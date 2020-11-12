from collections.abc import Iterable
import warnings
import dask
from dask.dataframe import DataFrame as DaskDataFrame
from dask.dataframe import from_delayed
import copy
from dask.base import is_dask_collection
from dask.distributed import Future

from .portsSpecSchema import PortsSpecSchema
from ._node import _Node

# OUTPUT_ID = 'f291b900-bd19-11e9-aca3-a81e84f29b0f_uni_output'
OUTPUT_ID = 'collector_id_fd9567b6'
OUTPUT_TYPE = 'Output_Collector'


__all__ = ['NodeTaskGraphMixin', 'OUTPUT_ID', 'OUTPUT_TYPE',
           'register_validator', 'register_copy_function']

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

_validators = {}  # dictionary of validators of funciton signiture (val, meta) -> bool

_copys = {}  # dictionary of object copy functions

_cleanup = {}  # dictionary of clean up functions


def _get_nodetype(node):
    '''Identify the implementation node class. A node might be mixed in with
    other classes. Ideally get the primary implementation class.
    '''
    nodetypes = node.__class__.mro()
    keeptypes = []
    for nodet in nodetypes:
        # Exclude base Node classes i.e. _Node, NodeTaskGraphMixin, Node.
        # Using nodet.__name__ != 'Node' to avoid cyclic dependencies.
        if issubclass(nodet, _Node) and \
                not issubclass(nodet, NodeTaskGraphMixin) and \
                nodet is not _Node and \
                nodet.__name__ != 'Node':
            keeptypes.append(nodet)

    return keeptypes


def register_validator(typename: type,
                       fun) -> None:
    # print('register validator for', typename)
    _validators[typename] = fun


def register_copy_function(typename: type,
                           fun) -> None:
    # print('register validator for', typename)
    _copys[typename] = fun


def register_cleanup(name: str,
                     fun) -> None:
    # print('register validator for', typename)
    _cleanup[name] = fun


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

        METHODS
        -------
            process
            load_cache
            save_cache
            _get_input_ports
            _get_output_ports
    '''

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'input_df' in state:
            del state['input_df']
        # print('state', state)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

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

    def __valide(self, node_output: dict):
        output_meta = self.meta_setup().outports
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

                # if self.delayed_process and \
                #         cudf.DataFrame in expected_type and \
                #         dask_cudf.DataFrame not in expected_type:
                #     expected_type.append(dask_cudf.DataFrame)

                match = False
                for expected in expected_type:
                    if issubclass(out_type, expected):
                        match = True
                        break

                if not match:
                    raise TypeError(
                        'Node "{}" output port "{}" produced wrong type '
                        '"{}". Expected type "{}"'
                        .format(self.uid, pname, out_type, expected_type))

            # cudf_types_tuple = (cudf.DataFrame, dask_cudf.DataFrame)
            # if out_type in cudf_types_tuple:
            #     if len(out_val.columns) == 0 and out_optional:
            #         continue

            if out_type in _validators:
                validator = _validators[out_type]
                meta_to_val = output_meta.get(pname)
                val_flag = validator(out_val, meta_to_val, self)
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

    def get_input_meta(self):
        """
        get all the connected input metas information
        returns
            dict, key is the current node input port name, value is the column
            name and types
        """
        output = {}
        if not hasattr(self, 'inputs'):
            return output
        for node_input in self.inputs:
            from_node = node_input['from_node']
            meta_data = copy.deepcopy(from_node.meta_setup())
            from_port_name = node_input['from_port']
            to_port_name = node_input['to_port']
            if from_port_name not in meta_data.outports:
                nodetype_list = _get_nodetype(self)
                nodetype_names = [inodet.__name__ for inodet in nodetype_list]
                if 'OutputCollector' in nodetype_names:
                    continue
                warnings.warn(
                    'node "{}" node-type "{}" to port "{}", from node "{}" '
                    'node-type "{}" oport "{}" missing oport in metadata for '
                    'node "{}" output meta: {}'.format(
                        self.uid, nodetype_list, to_port_name,
                        from_node.uid, _get_nodetype(from_node),
                        from_port_name, from_node.uid, meta_data.outports)
                )
            else:
                output[to_port_name] = meta_data.outports[from_port_name]
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
        typeObj = df_obj.__class__
        if typeObj in _copys:
            return _copys[typeObj](df_obj)
        else:
            return df_obj

    def __check_dly_processing_prereq(self, inputs: dict):
        '''At least one input must be a dask DataFrame type. Output types must
        be specified as cudf.DataFrame or dask_cudf.DataFrame. (Functionality
        could also be extended to support dask dataframe of pandas, but
        currently only cudf/dask_cudf dataframes are supported.)
        '''
        # check if dask future or delayed
        ivals = inputs.values()
        if not any((is_dask_collection(iv) for iv in ivals)) and \
                not any((isinstance(iv, Future) for iv in ivals)):
            # None of the inputs are Delayed or Futures so no intention of
            # using delayed processing. Return False and avoid printing
            # non-applicable warning.
            return False

        use_delayed = False
        for ival in ivals:
            if isinstance(ival, DaskDataFrame):
                use_delayed = True
                break

        # NOTE: Currently only support delayed processing when one of the
        #     inputs is a dask_cudf.DataFrame. In the future might generalize
        #     to support dask processing of other delayed/future type inputs.
        if not use_delayed:
            warn_msg = \
                'None of the Node "{}" inputs '\
                'is a dask_cudf.DataFrame. Ignoring '\
                '"delayed_process" setting.'.format(self.uid)
            warnings.warn(warn_msg)

        return use_delayed

    def __delayed_call(self, inputs: dict):
        '''Delayed processing called when self.delayed_process is set. To
        handle delayed processing automatically, prerequisites are checked via
        call to:
            :meth:`__check_dly_processing_prereq`
        Additionally all input dask_cudf dataframes have to be partitioned
        the same i.e. equal number of partitions.
        @param inputs: dict
            key: iport name string, value: input data
        '''

        def df_copy(df_in):
            '''Used for delayed unpacking.'''
            # Needed for the same reason as __make_copy. To prevent columns
            # addition in the input data frames. In python everything is
            # by reference value and dataframes are mutable.
            # Handle the case when dask_cudf.DataFrames are source frames
            # which appear as cudf.DataFrame in a dask-delayed function.
            return self.__make_copy(df_in)

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

            df_out = out_dict.get(port)

            return self.__make_copy(df_out)
        inputs_not_dly = {}
        for iport, inarg in inputs.items():
            # dcudf not necessarily a dask cudf frame
            if not isinstance(inarg, DaskDataFrame):
                # TODO: There could be cases where this non-delayed args are
                #     mutable. In that case USER BEWARE. Could copy here to
                #     deal with that. Shallow copy would be preferred but not
                #     100% reliable.
                inputs_not_dly[iport] = inarg

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
        for iport, inarg in inputs.items():
            # dcudf not necessarily a dask cudf frame
            if not isinstance(inarg, DaskDataFrame):
                continue
            dcudf = inarg
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
                # very import to use shallow copy of inputs_not_dly
                inputs_dly.setdefault(idly, inputs_not_dly.copy()).update({
                    # iport: dly.persist()  # DON'T PERSIST HERE
                    iport: dask.delayed(df_copy)(dly)
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
                oport_out = dask.delayed(get_pout)(output_df_dly_per, oport)
                outputs_dly.setdefault(oport, []).append(oport_out.persist())

        # DEBUGGING
        # print('OUTPUTS_DLY:\n{}'.format(outputs_dly))

        output_df = {}
        # A dask_cudf object is synthesized from a list of delayed objects.
        # Per outputs_dly above use dask_cudf.from_delayed API.
        for oport, port_spec in \
                self._get_output_ports(full_port_spec=True).items():
            port_type = port_spec.get(PortsSpecSchema.port_type, type(None))
            if not isinstance(port_type, Iterable):
                port_type = [port_type]
            # DEBUGGING
            # print('__DELAYED_CALL node "{}" port "{}" port type "{}"'.format(
            #     self.uid, oport, port_type))
            if any([issubclass(p_type, DaskDataFrame) for p_type in port_type]):
                output_df[oport] = from_delayed(outputs_dly[oport])
            else:
                # outputs_dly[oport] is currently a list. Run compute on each
                # partition, and keep the first one.
                # This is not very generalizeable
                # TODO: Check for equivalency and output a warning in case
                #     outputs don't match from different partitions.
                output_df[oport] = \
                    [iout.compute() for iout in outputs_dly[oport]][0]

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
            ports = from_node.calculated_ports_setup()
            from_port_name = node_input['from_port']
            to_port_name = node_input['to_port']
            if from_port_name in ports.outports:
                oport_types = get_type(
                    ports.outports[from_port_name][PortsSpecSchema.port_type])
                output[to_port_name] = oport_types
            else:
                continue
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
            self.__valide(output_df)

        if self.save:
            self.save_cache(output_df)

        return output_df
