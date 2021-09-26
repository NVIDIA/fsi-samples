from collections.abc import Iterable
import warnings
import dask
from dask.dataframe import DataFrame as DaskDataFrame
from dask.dataframe import from_delayed
import copy
from dask.base import is_dask_collection
from dask.distributed import Future

from .portsSpecSchema import PortsSpecSchema
from .taskSpecSchema import TaskSpecSchema
from .metaSpec import MetaData
from ._node import _Node
from ._node_taskgraph_extension_mixin import NodeTaskGraphExtensionMixin
from .output_collector_node import OUTPUT_TYPE


__all__ = ['NodeTaskGraphMixin', 'register_validator',
           'register_copy_function', 'register_cleanup']

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

# dictionary of validators of funciton signiture (val, meta) -> bool
_VALIDATORS = {}
# dictionary of object copy functions
_COPYS = {}
# dictionary of clean up functions
_CLEANUP = {}


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
                not issubclass(nodet, NodeTaskGraphExtensionMixin) and \
                nodet is not _Node and \
                nodet.__name__ != 'Node':
            keeptypes.append(nodet)

    return keeptypes


def register_validator(typename: type,
                       fun) -> None:
    # print('register validator for', typename)
    _VALIDATORS[typename] = fun


def register_copy_function(typename: type,
                           fun) -> None:
    # print('register validator for', typename)
    _COPYS[typename] = fun


def register_cleanup(name: str,
                     fun) -> None:
    # print('register validator for', typename)
    _CLEANUP[name] = fun


class NodeTaskGraphMixin(NodeTaskGraphExtensionMixin):
    '''Relies on mixing in with a Node class that has the following attributes
    and methods:
        ATTRIBUTES
        ----------
            _task_obj
            uid
            node_type_str
            conf
            load
            save
            delayed_process
            infer_meta

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

    def __getitem__(self, key):
        return getattr(self, key)

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

    def update(self):
        """
        Within a task graph context override a Node's update to cache: ports,
        metadata, input connections, and input metadata.
        """

        # this will filter out the primary class with ports_setup
        nodecls_list = _get_nodetype(self)
        for icls in nodecls_list:
            if not hasattr(icls, 'update'):
                continue
            # Within update resolve ports and meta via class NodeExtensionMixin
            # methods _resolve_ports and _resolve_meta.
            icls.update(self)
            break

        # cache it after update
        NodeTaskGraphExtensionMixin.cache_update_result(self)

        # cache the conf_schema too
        for icls in nodecls_list:
            if not hasattr(icls, 'conf_schema'):
                continue
            self.conf_schema_cache = self.conf_schema()
            break
        else:
            pass

    def conf_schema(self):
        if hasattr(self, 'conf_schema_cache'):
            return self.conf_schema_cache
        nodecls_list = _get_nodetype(self)
        for icls in nodecls_list:
            if not hasattr(icls, 'conf_schema'):
                continue
            return icls.conf_schema(self)

    def meta_setup(self):
        if hasattr(self, 'meta_data_cache'):
            return self.meta_data_cache

        nodecls_list = _get_nodetype(self)
        meta = MetaData()
        for icls in nodecls_list:
            if not hasattr(icls, 'meta_setup'):
                continue

            meta = icls.meta_setup(self)
            break

        return meta

    def ports_setup(self):
        """
        overwrite the super class ports_setup so it can calculate the dynamic
        ports.
        If ports information is needed, ports_setup should be used
        :return: Node ports
        :rtype: NodePorts
        """
        if hasattr(self, 'ports_setup_cache'):
            ports = self.ports_setup_cache
        else:
            # this will filter out the primary class with ports_setup
            nodecls_list = _get_nodetype(self)
            for icls in nodecls_list:
                if not hasattr(icls, 'ports_setup'):
                    continue

                ports = icls.ports_setup(self)
                break
            else:
                raise Exception('ports_setup method missing')

        # note, currently can only handle one dynamic port per node
        port_type = PortsSpecSchema.port_type
        inports = ports.inports
        dy = PortsSpecSchema.dynamic
        for key in inports:
            if dy in inports[key] and inports[key][dy]:
                types = inports[key][port_type]
                break
        else:
            return ports

        if hasattr(self, 'inputs'):
            has_dynamic = False
            for inp in self.inputs:
                to_port = inp['to_port']
                if to_port in inports and not inports[to_port].get(dy, False):
                    # skip connected non dynamic ports
                    continue
                else:
                    has_dynamic = True

            if has_dynamic:
                connected_inports = self.get_connected_inports()
                for inp in self.inputs:
                    to_port = inp['to_port']
                    if to_port in inports and (not inports[to_port].get(
                            dy, False)):
                        # skip connected non dynamic ports
                        continue
                    else:
                        if to_port in connected_inports:
                            types = connected_inports[to_port]
                        inports[inp['from_node'].uid+'@'+inp['from_port']] = {
                            port_type: types, dy: True}

        return ports

    def __validate_output(self, node_output: dict):
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

            if out_type in _VALIDATORS:
                validator = _VALIDATORS[out_type]
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

    def get_input_meta(self, port_name=None):
        """
        if port_name is None, get all the connected input metas information
        returns
            dict, key is the current node input port name, value is the column
            name and types
        if port_name is not None, get meta data for the input port_name. If it
        doesn't exist, return None
        """
        if hasattr(self, 'input_meta_cache'):
            if port_name is None:
                return self.input_meta_cache
            elif port_name in self.input_meta_cache:
                return self.input_meta_cache[port_name]
            else:
                # Run the logic below to find the port
                # Warning: Something might not be right if this happens.
                #     Perhaps the cache needs to be reset.
                pass

        output = {}
        if not hasattr(self, 'inputs'):
            return output

        out_port_names = []
        to_port_names = []
        from_port_names = []
        meta_data_list = []
        for node_input in self.inputs:
            from_node = node_input['from_node']
            meta_data = copy.deepcopy(from_node.meta_setup())

            from_port_name = node_input['from_port']
            to_port_name = node_input['to_port']
            if port_name is not None and port_name == to_port_name:
                return meta_data.outports[from_port_name]

            if from_port_name not in meta_data.outports:
                if self.node_type_str == OUTPUT_TYPE:
                    continue

                nodetype_list = _get_nodetype(self)
                warnings.warn(
                    'node "{}" node-type "{}" to port "{}", from node "{}" '
                    'node-type "{}" oport "{}" missing oport in metadata for '
                    'node "{}" output meta: {}'.format(
                        self.uid, nodetype_list, to_port_name,
                        from_node.uid, _get_nodetype(from_node),
                        from_port_name, from_node.uid, meta_data.outports)
                )
            else:
                out_port_name = from_node.uid+'@'+from_port_name
                out_port_names.append(out_port_name)
                to_port_names.append(to_port_name)
                from_port_names.append(from_port_name)
                meta_data_list.append(meta_data)

        if port_name is not None:
            return None

        if len(out_port_names) > 0:
            dy = PortsSpecSchema.dynamic
            ports = self.ports_setup()

            inports = ports.inports
            for out_port_name, to_port_name, from_port_name, meta_data in zip(
                    out_port_names, to_port_names, from_port_names,
                    meta_data_list):
                if out_port_name in inports and inports[out_port_name].get(
                        dy, False):
                    output[out_port_name] = meta_data.outports[from_port_name]
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

            # Prevent memory leaks.
            if not onode.visited:
                continue

            if oport is not None:
                if oport not in output_df:
                    if onode.node_type_str == OUTPUT_TYPE:
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
        if typeObj in _COPYS:
            return _COPYS[typeObj](df_obj)
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
            # output_df_dly_per = output_df_dly.persist()
            output_df_dly_per = output_df_dly
            for oport in self._get_output_ports():
                oport_out = dask.delayed(get_pout)(output_df_dly_per, oport)
                # outputs_dly.setdefault(oport, []).append(oport_out.persist())
                outputs_dly.setdefault(oport, []).append(oport_out)

        # DEBUGGING
        # print('OUTPUTS_DLY:\n{}'.format(outputs_dly))

        output_df = {}
        # A dask_cudf object is synthesized from a list of delayed objects.
        # Per outputs_dly above use dask_cudf.from_delayed API.
        connected_outs = set([out['from_port'] for out in self.outputs])
        for oport, port_spec in \
                self._get_output_ports(full_port_spec=True).items():
            if (oport not in connected_outs):
                continue
            port_type = port_spec.get(PortsSpecSchema.port_type, type(None))
            if not isinstance(port_type, Iterable):
                port_type = [port_type]
            # DEBUGGING
            # print('__DELAYED_CALL node "{}" port "{}" port type "{}"'.format(
            #     self.uid, oport, port_type))
            if any([issubclass(p_type,
                               DaskDataFrame) for p_type in port_type]):
                if self.infer_meta:
                    output_df[oport] = from_delayed(outputs_dly[oport])
                else:
                    meta_data = self.meta_setup().outports
                    output_df[oport] = from_delayed(outputs_dly[oport],
                                                    meta=meta_data[oport])
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

        if hasattr(self, 'input_connections_cache'):
            return self.input_connections_cache

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

        if self.node_type_str != OUTPUT_TYPE and output_df is None:
            raise Exception("None output")
        else:
            self.__validate_output(output_df)

        if self.save:
            self.save_cache(output_df)

        return output_df

    def _validate_connected_ports(self):
        """
        Validate the connected port types match
        """
        if self.node_type_str == OUTPUT_TYPE:
            # Don't validate for Output_Collector
            return

        self_nodetype = _get_nodetype(self)
        msgfmt = '"{task}":"{nodetype}" {inout} port "{ioport}" {inout} port '\
            'type(s) "{ioport_types}"'

        iports_connected = self.get_connected_inports()
        iports_spec = self._get_input_ports(full_port_spec=True)
        for iport in iports_connected.keys():
            iport_spec = iports_spec[iport]
            iport_type = iport_spec[PortsSpecSchema.port_type]
            iport_types = [iport_type] \
                if not isinstance(iport_type, Iterable) else iport_type

            for ient in self.inputs:
                # find input node edge entry with to_port, from_port, from_node
                if not iport == ient['to_port']:
                    continue
                ientnode = ient
                break
            else:
                intask = self._task_obj[TaskSpecSchema.inputs][iport]
                # this should never happen
                raise LookupError(
                    'Task "{}" not connected to "{}.{}". Add task spec to '
                    'TaskGraph.'.format(intask, self.uid, iport))

            from_node = ientnode['from_node']
            oport = ientnode['from_port']
            oports_spec = from_node._get_output_ports(full_port_spec=True)
            oport_spec = oports_spec[oport]
            oport_type = oport_spec[PortsSpecSchema.port_type]
            oport_types = [oport_type] \
                if not isinstance(oport_type, Iterable) else oport_type

            for optype in oport_types:
                if issubclass(optype, tuple(iport_types)):
                    break
            else:
                # Port types do not match
                msgi = msgfmt.format(
                    task=self.uid,
                    nodetype=self_nodetype,
                    inout='input',
                    ioport=iport,
                    ioport_types=iport_types)

                msgo = msgfmt.format(
                    task=from_node.uid,
                    nodetype=_get_nodetype(from_node),
                    inout='output',
                    ioport=oport,
                    ioport_types=oport_types)

                errmsg = 'Port Types Validation\n{}\n{}\n'\
                    'Connected nodes do not have matching port types. Fix '\
                    'port types.'.format(msgo, msgi)

                raise TypeError(errmsg)

    def _validate_connected_metadata(self):
        """
        Validate the connected metadata match the requirements.
        metadata.inports specify the required metadata.
        """
        metadata = self.meta_setup()

        # as current behavior of matching in the validate_required
        def validate_required(iport, kcol, kval, ientnode, icols):
            node = ientnode['from_node']
            oport = ientnode['from_port']
            src_task = '{}.{}'.format(node.uid, oport)
            src_type = _get_nodetype(node)
            # incoming "task.port":"Node-type":{{column:column-type}}
            msgi = \
                '"{task}":"{nodetype}" produces metadata {colinfo}'.format(
                    task=src_task,
                    nodetype=src_type,
                    colinfo=icols)

            dst_task = '{}.{}'.format(self.uid, iport)
            dst_type = _get_nodetype(self)
            # expecting "task.port":"Node-type":{{column:column-type}}
            msge = \
                '"{task}":"{nodetype}" requires metadata {colinfo}'.format(
                    task=dst_task,
                    nodetype=dst_type,
                    colinfo={kcol: kval})

            header = \
                'Meta Validation\n'\
                'Format "task.port":"Node-type":{{column:column-type}}'
            info_msg = '{}\n{}\n{}'.format(header, msgi, msge)

            if kcol not in icols:
                err_msg = \
                    'Task "{}" missing required column "{}" '\
                    'from "{}".'.format(self.uid, kcol, src_task)
                out_err = '{}\n{}'.format(info_msg, err_msg)
                raise LookupError(out_err)

            ival = icols[kcol]
            if kval != ival:
                if kval is None:
                    # bypass None type
                    return
                # special case for 'date'
                if (kval == 'date' and ival
                        in ('datetime64[ms]', 'date', 'datetime64[ns]')):
                    return
                else:
                    err_msg = 'Task "{}" column "{}" expected type "{}" got '\
                        'type "{}" instead.'.format(self.uid, kcol, kval, ival)
                    out_err = '{}\n{}'.format(info_msg, err_msg)
                    raise LookupError(out_err)

        inputs_meta = self.get_input_meta()
        required = metadata.inports

        if not required:
            return

        inports = self._get_input_ports(full_port_spec=True)
        for iport in inports:
            if iport not in required:
                continue
            required_iport = required[iport]

            if iport not in inputs_meta:
                # if iport is dynamic, skip warning
                dy = PortsSpecSchema.dynamic
                if inports[iport].get(dy, False):
                    continue

                if inports[iport].get(PortsSpecSchema.optional, False):
                    continue

                # Is it possible that iport not connected? If so iport should
                # not be in required. Should raise an exception here.
                warn_msg = \
                    'Task "{}" Node Type "{}" missing required port "{}" in '\
                    'incoming meta data. Should the port be connected?'.format(
                        self.uid, _get_nodetype(self), iport)
                warnings.warn(warn_msg)
                continue
            incoming_meta = inputs_meta[iport]

            for ient in self.inputs:
                # find input node edge entry with to_port, from_port, from_node
                if not iport == ient['to_port']:
                    continue
                ientnode = ient
                break
            else:
                intask = self._task_obj[TaskSpecSchema.inputs][iport]
                # this should never happen
                raise LookupError(
                    'Task "{}" not connected to "{}.{}". Add task spec to '
                    'TaskGraph.'.format(intask, self.uid, iport))

            for key, val in required_iport.items():
                validate_required(iport, key, val,
                                  ientnode, incoming_meta)
