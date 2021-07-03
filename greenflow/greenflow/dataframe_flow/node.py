
import abc
from .task import Task
from .taskSpecSchema import TaskSpecSchema
from .portsSpecSchema import (ConfSchema, NodePorts)
from .metaSpec import MetaData

from ._node import _Node
from ._node_extension_mixin import NodeExtensionMixin

__all__ = ['Node']


class _PortsMixin(object):
    '''Mixed class must have (doesn't have to implement i.e. relies on
    NotImplementedError) "ports_setup" method otherwise raises AttributeError.
    '''

    def __get_io_port(self, io=None, full_port_spec=False):
        input_ports, output_ports = self.ports_setup()
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


class Node(NodeExtensionMixin, _PortsMixin, _Node):
    '''Base class for implementing greenflow plugins i.e. nodes. A node
    processes tasks within a greenflow task graph.

    Within the context of a task graph the Node class is mixed in with classes
    NodeTaskGraphMixin and NodeTaskGraphExtensionMixin.

    Must implement the following method:

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
        node_type = task[TaskSpecSchema.node_type]
        self.node_type_str = node_type if isinstance(node_type, str) else \
             node_type.__name__
        self.conf = task[TaskSpecSchema.conf]
        self.load = task.get(TaskSpecSchema.load, False)
        self.save = task.get(TaskSpecSchema.save, False)

        self.delayed_process = False
        # eargerly infer the metadata, costly
        self.infer_meta = False
        # customized the column setup
        self.init()
        self.profile = False  # by default, do not profile

        # PortsSpecSchema.validate_ports(self.ports_setup())

    @abc.abstractmethod
    def ports_setup(self) -> NodePorts:
        """Virtual method for specifying inputs/outputs ports.

        Note: Within the context of a task graph the
            NodeTaskGraphMixin.ports_setup is invoked and forwards the call
            to the Node's implementation class ports_setup that is
            implementation of this method.

        Must return an instance of NodePorts that adheres to PortsSpecSchema.
        Refer to PortsSpecSchema and NodePorts in module:
            greenflow.dataframe_flow.portsSpecSchema

        Ex. ports for inputs and outputs. (typical case)
            inports = {
                'iport0_name': {
                    PortsSpecSchema.port_type: cudf.DataFrame
                },
                'iport1_name': {
                    PortsSpecSchema.port_type: cudf.DataFrame,
                    PortsSpecSchema.optional: True
                    PortsSpecSchema.dynamic: True
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

    def update(self):
        """
        Use the update method when relying on the dynamic information from
        the parent nodes.

        Call the self._resolve_ports and self._resolve_meta within update
        if using ports and meta templates. Refer to class NodeExtensionMixin
        and corresponding methods _resolve_ports and _resolve_meta.
        Refer to usage examples of class TemplateNodeMixin.update.

        """
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
        Initialize the node. Usually it is used to set self.delayed_process
        flag and other special initialzation.

        The self.delayed_process flag is by default set to False. It can be
        overwritten here to True. For native dataframe API calls, dask cudf
        support the distribution computation. But the dask dataframe does
        not support GPU customized kernels directly. We can use to_delayed and
        from_delayed low level interfaces of dask dataframe to add this
        support.  In order to use Dask (for distributed computation i.e.
        multi-gpu in examples later on) we set the flag and the framework
        handles dask dataframes automatically under the hood.
        """
        pass

    def get_input_meta(self, port_name=None):
        """
        Get the input meta information. It is usually used by individual
         node to compute the output meta information
        if port_name is None
        returns
            dict, key is the node input port name, value is the metadata dict
        if port_name is provided
        returns
            the meta data send to the input port with name `port_name`. If it
            is not connected, return None
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

        Note: Within the context of a task graph the
            NodeTaskGraphMixin.meta_setup is invoked and forwards the call
            to the Node's implementation class meta_setup that is
            implementation of this method.

        `inports` defines the required metadata for the node.
        metadata is python dictionaries, which can be serialized into JSON.
        The output metadata are calcuated based on the input meta data. It is
        passed to the downstream nodes to do metadata validation

        returns:
        :return: MetaData
        """
        return MetaData(inports={}, outports={})

    @abc.abstractmethod
    def process(self, inputs, **kwargs) -> dict:
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
    def validate_connected_ports(self) -> None:
        """
        Validate the connected port types match. If not overwritte, it uses
        the default implementation
        """
        if hasattr(self, '_validate_connected_ports'):
            self._validate_connected_ports()

    def validate_connected_metadata(self) -> None:
        """
        Validate the connected metadata match the requirements.
        metadata.inports specify the required metadata.

        If not overwritte, it uses the default implementation

        """
        if hasattr(self, '_validate_connected_metadata'):
            self._validate_connected_metadata()

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

    def save_cache(self, output_data: dict) -> None:
        '''Defines how to save the output of a node to
        filesystem cache.

        :param output_data: The output from :meth:`process`.
        '''
        raise NotImplementedError
