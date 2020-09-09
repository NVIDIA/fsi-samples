import warnings
import numpy as np
import pandas as pd
import dask
import cudf
import dask_cudf
import copy

from .taskSpecSchema import TaskSpecSchema
from .portsSpecSchema import PortsSpecSchema

# OUTPUT_ID = 'f291b900-bd19-11e9-aca3-a81e84f29b0f_uni_output'
OUTPUT_ID = 'collector_id_fd9567b6'
OUTPUT_TYPE = 'Output_Collector'


__all__ = ['NodeTaskGraphMixin', 'OUTPUT_ID', 'OUTPUT_TYPE']

# class NodeIncomingEdge(object):
#     from_node = 'from_node'
#     from_port = 'from_port'
#     to_node = 'to_port'
#
#
# class NodeOutgoingEdge(object):
#     to_node = 'to_node'
#     to_port = 'to_port'
#     from_port = 'from_port'


class NodeTaskGraphMixin(object):
    '''Relies on mixing in with a Node class that has the following attributes
    and methods:
        ATTRIBUTES
        ----------
            _task_obj
            uid
            conf
            load
            save
            delayed_process

            required
            addition
            deletion
            retention
            rename

        METHODS
        -------
            process
            load_cache
            save_cache
            _get_input_ports
            _get_output_ports
    '''

    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.visited = False

        self.input_df = {}
        # input_df format:
        # {
        #     iport0: df_for_iport0,
        #     iport1: df_for_iport1,
        # }
        # Note: that even though the "df" terminology is used the type is
        #     user configurable i.e. "df" is just some python object which is
        #     typically a data container.
        self.clear_input = True

    def __translate_column(self, columns):
        output = {}
        for col_name, col_type in columns.items():
            if col_type is not None and col_type.startswith("@"):
                col_type = self.conf[col_type[1:]]
            if col_name.startswith("@"):
                field_name = col_name[1:]
                v = self.conf[field_name]
                if isinstance(v, str):
                    output[v] = col_type
                elif isinstance(v, list):
                    for item in v:
                        output[item] = None
            else:
                output[col_name] = col_type

        return output

    def __delayed_call_noports(self, inputs):

        def get_pout(df_out):
            '''Used for delayed unpacking.'''
            if isinstance(df_out, cudf.DataFrame):
                # Needed for the same reason as __make_copy. To prevent columns
                # addition in the input data frames. In python everything is
                # by reference value and dataframes are mutable.
                # Handle the case when dask_cudf.DataFrames are source frames
                # which appear as cudf.DataFrame in a dask-delayed function.
                return df_out.copy(deep=False)

            return df_out

        # handle the dask dataframe automatically
        # use the to_delayed interface
        # TODO, currently only handles first input is dask_cudf df
        i_df = inputs[0]
        rest = inputs[1:]
        if isinstance(i_df, dask_cudf.DataFrame):
            output_df_dly_list = []
            for input_dly in i_df.to_delayed():
                inputs_ = [input_dly] + rest
                output_df_dly = dask.delayed(self.decorate_process())(inputs_)
                output_df_dly_per = output_df_dly.persist()
                df_out = dask.delayed(get_pout)(output_df_dly_per)
                output_df_dly_list.append(df_out.persist())

            output_df = dask_cudf.from_delayed(output_df_dly_list)

        else:
            output_df = self.decorate_process()(inputs)

        return output_df

    def validate_required_columns(self):

        def validate_required(icols, kcol, kval, in_taskid=None, iport=None):
            if kcol not in icols:
                err_msg = 'Incoming columns not valid: error for node "%s", '\
                    'missing required column "%s".' % (self.uid, kcol)
                if in_taskid:
                    dst_uid = self.uid if iport is None else \
                        '{}.{}'.format(self.uid, iport)
                    err_msg = '{}\nIncoming columns from "{}" do not match '\
                        'columns_setup for "{}".'.format(
                            err_msg, in_taskid, dst_uid)
                # raise Exception(err_msg)
                print(err_msg)
            if (kcol in icols) and kval != icols[kcol]:
                # special case for 'date'
                if (kval == 'date' and icols[kcol]
                        in ('datetime64[ms]', 'date', 'datetime64[ns]')):
                    # continue
                    return
                else:
                    print("error for node %s, "
                          "type %s mismatch %s"
                          % (self.uid, kval, icols[kcol]))

        inputs_cols = self.get_input_columns()

        # to_port (iport usually used as variable) is always set. Refer to
        # TaskGraph.build method. In non-port case inputs are enumerated
        # in the order that inputs are listed in the task spec. The order
        # idx is used as an ad-hoc ports that aren't used. Below the data
        # structure of inputs_cols is flattened.
        incoming_cols = {
            col_name: col_type for icol_dict in inputs_cols.values()
            for col_name, col_type in icol_dict.items()
        }
        inputs_cols = incoming_cols

        if self.required:
            required = self.required
            pinputs = self._task_obj[TaskSpecSchema.inputs]
            for iport in self._get_input_ports():
                required_iport = {
                    col_name: col_type for col_name, col_type in
                    required.get(iport, {}).items()}

                # required_tran = self.__translate_column(required_iport)
                required_tran = required_iport
                if iport in inputs_cols:
                    incoming_cols = inputs_cols[iport]
                    in_taskid = pinputs[iport]

                    for kcol, kval in required_tran.items():
                        validate_required(incoming_cols, kcol, kval,
                                          in_taskid, iport)

    def get_output_columns(self):
        return self.columns_setup()

    def _validate_df(self, df_to_val, ref_cols):
        '''Validate a cudf or dask_cudf DataFrame.

        :param df_to_val: A dataframe typically of type cudf.DataFrame or
            dask_cudf.DataFrame.
        :param ref_cols: Dictionary of column names and their expected types.
        :returns: True or False based on matching all columns in the df_to_val
            and columns spec in ref_cols.
        :raises: Exception - Raised when invalid dataframe length or unexpected
            number of columns. TODO: Create a ValidationError subclass.

        '''
        if (isinstance(df_to_val, cudf.DataFrame) or
            isinstance(df_to_val, dask_cudf.DataFrame)) and \
                len(df_to_val) == 0:
            err_msg = 'Node "{}" produced empty output'.format(self.uid)
            raise Exception(err_msg)

        if not isinstance(df_to_val, cudf.DataFrame) and \
           not isinstance(df_to_val, dask_cudf.DataFrame):
            return True

        i_cols = df_to_val.columns
        if len(i_cols) != len(ref_cols):
            print("expect %d columns, only see %d columns"
                  % (len(ref_cols), len(i_cols)))
            print("ref:", ref_cols)
            print("columns", i_cols)
            raise Exception("not valid for node %s" % (self.uid))

        for col in ref_cols.keys():
            if col not in i_cols:
                print("error for node %s, column %s is not in the required "
                      "output df" % (self.uid, col))
                return False

            if ref_cols[col] is None:
                continue

            err_msg = "for node {} type {}, column {} type {} "\
                "does not match expected type {}".format(
                    self.uid, type(self), col, df_to_val[col].dtype,
                    ref_cols[col])

            if ref_cols[col] == 'category':
                # comparing pandas.core.dtypes.dtypes.CategoricalDtype to
                # numpy.dtype causes TypeError. Instead, let's compare
                # after converting all types to their string representation
                # d_type_tuple = (pd.core.dtypes.dtypes.CategoricalDtype(),)
                d_type_tuple = (str(pd.CategoricalDtype()),)
            elif ref_cols[col] == 'date':
                # Cudf read_csv doesn't understand 'datetime64[ms]' even
                # though it reads the data in as 'datetime64[ms]', but
                # expects 'date' as dtype specified passed to read_csv.
                d_type_tuple = ('datetime64[ms]', 'date', 'datetime64[ns]')
            else:
                d_type_tuple = (str(np.dtype(ref_cols[col])),)

            if (str(df_to_val[col].dtype) not in d_type_tuple):
                print("ERROR: {}".format(err_msg))
                # Maybe raise an exception here and have the caller
                # try/except the validation routine.
                return False

        return True

    def __valide(self, node_output, ref_cols):
        # Validate each port
        out_ports = self._get_output_ports(full_port_spec=True)
        for pname, pspec in out_ports.items():
            # only validate it if it is connected
            if not self.outport_connected(pname):
                # if the port is not connected skip it
                # print('port {} is not connected'.format(pname))
                continue
            out_optional = pspec.get('optional', False)
            if pname not in node_output:
                if out_optional:
                    continue
                else:
                    raise Exception('Node "{}" did not produce output "{}"'
                                    .format(self.uid, pname))

            out_val = node_output[pname]
            out_type = type(out_val)

            expected_type = pspec.get(PortsSpecSchema.port_type)
            if expected_type:
                if not isinstance(expected_type, list):
                    expected_type = [expected_type]

                if self.delayed_process and \
                        cudf.DataFrame in expected_type and \
                        dask_cudf.DataFrame not in expected_type:
                    expected_type.extend([dask_cudf.DataFrame])

                if out_type not in expected_type:
                    raise Exception(
                        'Node "{}" output port "{}" produced wrong type '
                        '"{}". Expected type "{}"'
                        .format(self.uid, pname, out_type, expected_type))

            cudf_types_tuple = (cudf.DataFrame, dask_cudf.DataFrame)

            if out_type in cudf_types_tuple:
                if len(out_val.columns) == 0 and out_optional:
                    continue

            if out_type in cudf_types_tuple:
                cols_to_val = ref_cols.get(pname)
                val_flag = self._validate_df(out_val, cols_to_val)
                if not val_flag:
                    raise Exception("not valid output")

    def __input_ready(self):
        if not isinstance(self.load, bool) or self.load:
            return True

        for ient in self.inputs:
            iport = ient['to_port']

            if iport not in self.input_df:
                return False

        return True

    def __get_input_df(self):
        return self.input_df

    def get_input_columns(self):
        """
        get all the connected input columns information
        returns
            dict, key is the current node input port name, value is the column
            name and types
        """
        output = {}
        if not hasattr(self, 'inputs'):
            return output
        for node_input in self.inputs:
            from_node = node_input['from_node']
            columns = copy.deepcopy(from_node.columns_setup())
            from_port_name = node_input['from_port']
            to_port_name = node_input['to_port']
            if from_port_name not in columns:
                print('node {},  from {} oport {} not in out cols'.format(
                    self.uid, from_node, from_port_name))
                print(columns)
            else:
                output[to_port_name] = columns[from_port_name]
        return output

    def __set_input_df(self, to_port, df):
        self.input_df[to_port] = df

    def flow(self, progress_fun=None):
        """
        progress_fun is used to show the progress of computaion
        it is function that takes node id as argument
        flow from this node to do computation.
            * it will check all the input dataframe are ready or not
            * calls its process function to manipulate the input dataframes
            * set the resulting dataframe to the children nodes as inputs
            * flow each of the chidren nodes
        """
        if progress_fun is not None:
            progress_fun(self.uid)
        input_ready = self.__input_ready()
        if not input_ready:
            return

        inputs_data = self.__get_input_df()
        output_df = self.__call__(inputs_data)

        if self.clear_input:
            self.input_df = {}

        for out in self.outputs:
            onode = out['to_node']
            iport = out['to_port']
            oport = out['from_port']

            if oport is not None:
                if oport not in output_df:
                    if onode.uid in (OUTPUT_ID,):
                        onode_msg = 'is listed in task-graph outputs'
                    else:
                        onode_msg = 'is required as input to node "{}"'.format(
                            onode.uid)
                    err_msg = 'ERROR: Missing output port "{}" from '\
                        'node "{}". This output {}.'.format(
                            oport, self.uid, onode_msg)
                    raise Exception(err_msg)
                df = output_df[oport]

            onode.__set_input_df(iport, df)

            if onode.visited:
                onode.flow(progress_fun)

    def __make_copy(self, df_obj):
        if isinstance(df_obj, cudf.DataFrame):
            return df_obj.copy(deep=False)
        elif isinstance(df_obj, dask_cudf.DataFrame):
            # TODO: This just makes a df_obj with a shallow copy of the
            #     underlying computational graph. It does not affect the
            #     underlying data. Why is a copy of dask graph needed?
            return df_obj.copy()
        else:
            return df_obj

    def __check_dly_processing_prereq(self, inputs):
        '''All inputs must be dask_cudf.DataFrame types. Output types must
        be specified as cudf.DataFrame or dask_cudf.DataFrame. (Functionality
        could also be extended to support dask.dataframe.DataFrame, but
        currently only cudf/dask_cudf dataframes are supported.)
        '''
        # check if dask future or delayed
        use_delayed = False
        in_types = {}
        for iport, ival in inputs.items():
            itype = type(ival)
            in_types[iport] = itype
            if itype in (dask_cudf.DataFrame,):
                use_delayed = True

        if use_delayed:
            warn_msg = \
                'Node "{}" iport "{}" is of type "{}" and it '\
                'should be dask_cudf.DataFrame. Ignoring '\
                '"delayed_process" setting.'
            for iport, itype in in_types.items():
                if itype not in (dask_cudf.DataFrame,):
                    warnings.warn(warn_msg.format(self.uid, iport, itype))
                    use_delayed = False

        if use_delayed:
            warn_msg = \
                'Node "{}" oport "{}" is of type "{}" and it '\
                'should be cudf.DataFrame or dask_cudf.DataFrame. Ignoring '\
                '"delayed_process" setting.'
            for oport, oport_spec in \
                    self._get_output_ports(full_port_spec=True).items():
                otype = oport_spec.get('type', [])
                if not isinstance(otype, list):
                    otype = [otype]
                if dask_cudf.DataFrame not in otype and \
                        cudf.DataFrame not in otype:
                    warnings.warn(warn_msg.format(self.uid, oport, otype))
                    use_delayed = False

        return use_delayed

    def __delayed_call(self, inputs):
        '''Delayed processing called when self.delayed_process is set. To
        handle delayed processing automatically, prerequisites are checked via
        call to:
            :meth:`__check_dly_processing_prereq`
        Additionally all input dask_cudf dataframes have to be partitioned
        the same i.e. equal number of partitions.
        '''

        def get_pout(out_dict, port):
            '''Get the output in out_dict at key port. Used for delayed
            unpacking.'''
            # DEBUGGING
            # try:
            #     from dask.distributed import get_worker
            #     worker = get_worker()
            #     print('worker{} get_pout NODE "{}" port "{}" worker: {}'
            #           .format(worker.name, self.uid, port, worker))
            # except Exception as err:
            #     print(err)

            df_out = out_dict.get(port, cudf.DataFrame())

            if isinstance(df_out, cudf.DataFrame):
                # Needed for the same reason as __make_copy. To prevent columns
                # addition in the input data frames. In python everything is
                # by reference value and dataframes are mutable.
                # Handle the case when dask_cudf.DataFrames are source frames
                # which appear as cudf.DataFrame in a dask-delayed function.
                return df_out.copy(deep=False)

            return df_out

        inputs_dly = {}
        # A dask_cudf object will return a list of dask delayed object using
        # to_delayed() API. Below the logic assumes (otherwise error) that
        # all inputs are dask_cudf objects and are distributed in the same
        # manner. Ex. inputs_dly:
        #     inputs_dly = {
        #         p0: {
        #             iport0: ddf_dly_i0_p0,
        #             iport1: ddf_dly_i1_p0,
        #             ... for all iports
        #         },
        #         p1: {
        #             iport0: ddf_dly_i0_p1,
        #             iport1: ddf_dly_i1_p1,
        #             ... for all iports
        #         },
        #         ... for all partitions
        # i_x - iport
        # p_x - partition index

        npartitions = None
        for iport, dcudf in inputs.items():
            ddf_dly_list = dcudf.to_delayed()
            npartitions_ = len(ddf_dly_list)
            if npartitions is None:
                npartitions = npartitions_
            if npartitions != npartitions_:
                raise Exception(
                    'Error DASK_CUDF PARTITIONS MISMATCH: Node "{}" input "{}"'
                    ' has {} npartitions and other inputs have {} partitions'
                    .format(self.uid, iport, npartitions_, npartitions))
            for idly, dly in enumerate(ddf_dly_list):
                inputs_dly.setdefault(idly, {}).update({
                    # iport: dly.persist()  # DON'T PERSIST HERE
                    iport: dly
                })

        # DEBUGGING
        # print('INPUTS_DLY:\n{}'.format(inputs_dly))

        outputs_dly = {}
        # Formulate a list of delayed objects for each output port to be able
        # to call from_delayed to synthesize a dask_cudf object.
        # Ex. outputs_dly:
        #     outputs_dly = {
        #         o0: [ddf_dly_o0_p0, ddf_dly_o0_p1, ... _pN]
        #         o1: [ddf_dly_o1_p0, ddf_dly_o1_p1, ... _pN]
        #         ... for all output ports
        #     }
        # o_x - output port
        # p_x - delayed partition

        # VERY IMPORTANT TO USE PERSIST:
        # https://docs.dask.org/en/latest/dataframe-api.html#dask.dataframe.DataFrame.persist
        # Otherwise process will run several times.
        for inputs_ in inputs_dly.values():
            output_df_dly = dask.delayed(self.decorate_process())(inputs_)
            output_df_dly_per = output_df_dly.persist()
            for oport in self._get_output_ports():
                oport_out = dask.delayed(get_pout)(
                    output_df_dly_per, oport)
                outputs_dly.setdefault(oport, []).append(oport_out.persist())

        # DEBUGGING
        # print('OUTPUTS_DLY:\n{}'.format(outputs_dly))

        output_df = {}
        # A dask_cudf object is synthesized from a list of delayed objects.
        # Per outputs_dly above use dask_cudf.from_delayed API.
        for oport in self._get_output_ports():
            output_df[oport] = dask_cudf.from_delayed(outputs_dly[oport])

        return output_df

    def outport_connected(self, port_name):
        """
        test whether this node's output port is connected.
        @params port_name
            string, outpout port name
        returns
            boolean, whehther this port is connected or not
        """
        found = False
        for iout in self.outputs:
            oport = iout['from_port']
            if (port_name == oport):
                found = True
                break
        return found

    def get_connected_inports(self):
        """
        get all the connected input port information
        returns
            dict, key is the current node input port name, value is the port
            type passed from parent
        """

        def get_type(type_def):
            if isinstance(type_def, list):
                return type_def
            else:
                return [type_def]
        output = {}
        if not hasattr(self, 'inputs'):
            return output
        for node_input in self.inputs:
            from_node = node_input['from_node']
            ports = from_node.ports_setup()
            from_port_name = node_input['from_port']
            to_port_name = node_input['to_port']
            types = get_type(ports.outports[from_port_name]['type'])
            output[to_port_name] = types
        return output

    def decorate_process(self):

        def timer(*argv):
            import time
            start = time.time()
            result = self.process(*argv)
            end = time.time()
            print('id:%s process time:%.3fs' % (self.uid, end-start))
            return result
        if self.profile:
            return timer
        else:
            return self.process

    def __call__(self, inputs_data):
        if self.load:
            if isinstance(self.load, bool):
                output_df = self.load_cache()
            else:
                output_df = self.load
        else:
            # nodes with ports take dictionary as inputs
            inputs = {iport: self.__make_copy(data_input)
                      for iport, data_input in inputs_data.items()}
            if not self.delayed_process:
                output_df = self.decorate_process()(inputs)
            else:
                use_delayed = self.__check_dly_processing_prereq(inputs)
                if use_delayed:
                    output_df = self.__delayed_call(inputs)
                else:
                    output_df = self.decorate_process()(inputs)

        if self.uid != OUTPUT_ID and output_df is None:
            raise Exception("None output")
        else:
            self.__valide(output_df, self.columns_setup())

        if self.save:
            self.save_cache(output_df)

        return output_df
