import warnings
import abc
from collections.abc import Iterable
from .task import Task
from .taskSpecSchema import TaskSpecSchema
from .portsSpecSchema import PortsSpecSchema, ConfSchema, MetaData, NodePorts
from ._node_flow import _get_nodetype

from ._node import _Node


__all__ = ['Node']


class _PortsMixin(object):
    '''Mixed class must have (doesn't have to implement i.e. relies on
    NotImplementedError) "ports_setup" method otherwise raises AttributeError.
    '''

    def __get_io_port(self, io=None, full_port_spec=False):
        input_ports, output_ports = self.calculated_ports_setup()
        if io in ('in',):
            io_ports = input_ports
        else:
            io_ports = output_ports

        if io_ports is None:
            io_ports = dict()

        if not full_port_spec:
            io_ports = list(io_ports.keys())

        return io_ports

    def _get_input_ports(self, full_port_spec=False):
        return self.__get_io_port(io='in', full_port_spec=full_port_spec)

    def _get_output_ports(self, full_port_spec=False):
        return self.__get_io_port(io='out', full_port_spec=full_port_spec)


class Node(_PortsMixin, _Node):
    '''Base class for implementing gQuant plugins i.e. nodes. A node processes
    tasks within a gQuant task graph.

    must implement the following method:

        :meth: ports_setup
            Defines ports for the node. Refer to ports_setup docstring for
            further details.

        :meth: meta_setup
            Define expected metadata and resulting metadata.
            Ex. When processing dataframes define expected columns:
                def meta_setup(self):
                    metadata = MetaData()
                    required = {
                        'iport0_name': {'x': 'float64',
                                        'y': 'float64'}
                        'iport1_name': some_dict,
                        etc.
                    }
                    out_cols = {
                        'oport0_name': {'x': 'float64',
                                        'y': 'float64',
                                        'z': 'int64'}
                        'oport1_name': some_dict,
                        etc.
                    }
                    metadata.inports = required
                    metadata.outports = out_cols
                    return metadata
            Refer to meta_setup docstring for further details.

        :meth: conf_schema
            Define the json schema for the Node configuration. The client
            can automatically generate the UI elements based on the schema
            Refer to process docstring for further details.

        :meth: process
            Main functionaliy or processing logic of the Node. Refer to
            process docstring for further details.

    '''

    cache_dir = '.cache'

    def __init__(self, task):
        # make sure is is a task object
        assert isinstance(task, Task)
        self._task_obj = task  # save the task obj
        self.uid = task[TaskSpecSchema.task_id]
        self.conf = task[TaskSpecSchema.conf]
        self.load = task.get(TaskSpecSchema.load, False)
        self.save = task.get(TaskSpecSchema.save, False)

        self.delayed_process = False
        # customized the column setup
        self.init()
        self.profile = False  # by default, do not profile

        PortsSpecSchema.validate_ports(self.ports_setup())

    def calculated_ports_setup(self):
        # note, currently can only handle one dynamic port per node
        port_type = PortsSpecSchema.port_type
        ports = self.ports_setup()
        inports = ports.inports
        dy = PortsSpecSchema.dynamic
        for key in inports:
            if dy in inports[key] and inports[key][dy]:
                types = inports[key][port_type]
                break
        else:
            return ports
        if hasattr(self, 'inputs'):
            for inp in self.inputs:
                to_port = inp['to_port']
                if to_port in inports and (not inports[to_port].get(dy,
                                                                    False)):
                    # skip connected non dynamic ports
                    continue
                else:
                    inports[inp['from_node'].uid+'@'+inp['from_port']] = {
                        port_type: types, dy: True}
        return ports

    def ports_setup(self) -> NodePorts:
        """Virtual method for specifying inputs/outputs ports.

        Must return an instance of NodePorts that adheres to PortsSpecSchema.
        Refer to PortsSpecSchema and NodePorts in module:
            gquant.dataframe_flow.portsSpecSchema

        Ex. ports for inputs and outputs. (typical case)
            inports = {
                'iport0_name': {
                    PortsSpecSchema.port_type: cudf.DataFrame
                },
                'iport1_name': {
                    PortsSpecSchema.port_type: cudf.DataFrame,
                    PortsSpecSchema.optional: True
                }
            }

            outports = {
                'oport0_name': {
                    PortsSpecSchema.port_type: cudf.DataFrame
                },
                'oport1_name': {
                    PortsSpecSchema.port_type: cudf.DataFrame,
                    PortsSpecSchema.optional: True
                }
            }

            node_ports = NodePorts(inports=inports, outports=outports)
            return node_ports

        The output port type can be dynamically calculated based on the input
        port types. The input port type can be obtained by
        `self.get_connected_inports` method.

        :return: Node ports
        :rtype: NodePorts

        """
        raise NotImplementedError

    def outport_connected(self, portname) -> bool:
        """
        Test whether this node's output port is connected. It is used
        to generate result for the output port based on the connection
        condition
        @params port_name
            string, outpout port name
        returns
            boolean, whehther this port is connected or not
        """
        # this method will be implemented by NodeTaskGraphMixin
        pass

    def get_connected_inports(self) -> dict:
        """
        Get all the connected input port information. It is used by individual
        node to determine the output port types
        returns
            dict, key is the current node input port name, value is the port
            type passed from parent
        """
        # this method will be implemented by NodeTaskGraphMixin
        return {}

    def init(self):
        """
        Initialize the node. Usually it is used to self.delayed_process flag
         and other special initialzation.

        The self.delayed_process flag is by default set to False. It can be
        overwritten here to True. For native dataframe API calls, dask cudf
        support the distribution computation. But the dask_cudf dataframe does
        not support GPU customized kernels directly. We can use to_delayed and
        from_delayed low level interfaces of dask_cudf to add this support.
        In order to use Dask (for distributed computation i.e. multi-gpu in
        examples later on) we set the flag and the framework
        handles dask_cudf dataframes automatically under the hood.
        """
        pass

    def get_input_meta(self) -> dict:
        """
        Get the input meta information. It is usually used by individual
         node to compute the output meta information
        returns
            dict, key is the node input port name, value is the metadata dict
        """
        # this method will be implemented by NodeTaskGraphMixin
        return {}

    def conf_schema(self) -> ConfSchema:
        """Virtual method for specifying configuration schema. Implement if
        desire to use the UI client to help fill the conf forms.

        The schema standard is specified by
        [JSON Schema](https://json-schema.org/)

        The UI Client side uses this [open source React component]
        (https://github.com/rjsf-team/react-jsonschema-form)

        To learn how to write JSON schema, please refer to this [document]
        (https://react-jsonschema-form.readthedocs.io/en/latest/).

        :return: Conf Schema
        :rtype: ConfSchema

        """
        return ConfSchema(json={}, ui={})

    @abc.abstractmethod
    def meta_setup(self) -> MetaData:
        """
        All children class should implement this.
        It is used to compute the required input and output meta data.

        `inputs` defines the required metadata 
        metadata is python dictionaries, which can be serialized into JSON
      
        The output metadata are calcuated based on the input meta data. It is 
        passed to the downstream nodes to do metadata validation

        returns:
        :return: MetaData
        """
        return MetaData(inports={}, outports={})

    @abc.abstractmethod
    def process(self, inputs) -> dict:
        """
        process the input dataframe. Children class is required to override
        this

        Arguments
        -------
        inputs: dictionary
            the inputs is a dictionary keyed by port name as defined in 
            ports_setup.
                Ex.:
                    inputs = {
                        iport0: df0,
                        iport1: df1,
                        etc.
                    }
                The task-spec for inputs is a dictionary keyed by port names
                 with values being task-ids of input tasks "." port output of 
                 the input tasks. Ex.:
                    TaskSpecSchema.inputs: {
                        iport0: some_task_id.some_oport,
                        iport1: some_other_task_id.some_oport,
                        etc.
                    }
                Within the process access the dataframes (data inputs) as:
                    df0 = inputs[iport0] # from some_task_id some_oport
                    df1 = inputs[iport1] # from some_other_task_id some_oport
                    etc.

        Returns
        -------
        dataframe
            The output can be anything representable in python. Typically it's
            a processed dataframe.

            It return a dictionary keyed by output ports (as defined in 
            ports_setup). Ex.:
                df = cudf.DataFrame()  # or maybe it can from an input
                # do some calculations and populate df.
                return {oport: df}

            If there are mutliple output ports, the computation can be done
            on demand depending on whether the output port is connected or
            not. The output connection can be queried by
            `self.outport_connected` method
        """
        output = None
        return output

# Validation methods ######################################
    def validate_connected_ports(self):
        """
        Validate the connected port types match
        """
        self_nodetype = _get_nodetype(self)
        nodetype_names = [inodet.__name__ for inodet in self_nodetype]
        if 'OutputCollector' in nodetype_names:
            # Don't validate for OutputCollector
            return

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

    def validate_connected_metadata(self):
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
        inports = self._get_input_ports()
        for iport in inports:
            if iport not in required:
                continue
            required_iport = required[iport]

            if iport not in inputs_meta:
                # if iport is dynamic, skip warning
                dy = PortsSpecSchema.dynamic
                if inports[iport].get(dy, False):
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

    def load_cache(self, filename=None) -> dict:
        """
        Defines the behavior of how to load the cache file from the `filename`.
        Node can override this method.         Arguments
        -------
        filename: str
            filename of the cache file. Leave as none to use default.
        returns: dict
            dictionary of the output from this node
        """
        raise NotImplementedError

    def save_cache(self, output_data: dict):
        '''Defines how to save the output of a node to
        filesystem cache.

        :param output_data: The output from :meth:`process`.
        '''
        raise NotImplementedError
