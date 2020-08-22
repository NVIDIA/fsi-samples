import os
import warnings
import abc
import pandas as pd
import cudf

from .task import Task
from .taskSpecSchema import TaskSpecSchema
from .portsSpecSchema import PortsSpecSchema, ConfSchema

from ._node import _Node


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


class Node(_PortsMixin, _Node):
    '''Base class for implementing gQuant plugins i.e. nodes. A node processes
    tasks within a gQuant task graph.

    must implement the following method:

        :meth: ports_setup
            Defines ports for the node. Refer to ports_setup docstring for
            further details.

        :meth: columns_setup
            Define expected columns in dataframe processing.
            When processing dataframes define expected columns. Ex.:
                def columns_setup(self):
                    self.required = {
                        'iport0_name': {'x': 'float64',
                                        'y': 'float64'}
                        'iport1_name': some_dict,
                        etc.
                    }
            Refer to columns_setup docstring for further details.

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

        self.required = {}
        self.delayed_process = False
        # customized the column setup
        self.init()
        self.profile = False  # by default, do not profile

        PortsSpecSchema.validate_ports(self.ports_setup())

    def ports_setup(self):
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

    def outport_connected(self, portname):
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

    def get_connected_inports(self):
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

    def get_input_columns(self):
        """
        Get the input column information. It is usually used by individual
         node to compute the output column information
        returns
            dict, key is the node input port name, value is the dict with keys
            column names, and values column types
        """
        # this method will be implemented by NodeTaskGraphMixin
        return {}

    def conf_schema(self):
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
    def columns_setup(self):
        """
        All children class should implement this.
        It is used to compute the required and output column names and types.

        `self.required` defines the required columns in the input dataframes
        `self.required` is python dictionaries, where keys are column names
         and values are column types.
       
        Example column types:
            * int64
            * int32
            * float64
            * float32
            * datetime64[ms]

        The output columns are calcuated based on the input columns. The input
        column name and types can be obtained by `self.get_input_columns` method.

        returns:
            dict, key is the node output port name, value is the dict with keys
            column names, and values column types
        """
        self.required = {}
        self.addition = {}
        self.deletion = {}
        # Retention must be None instead of empty dict. This replaces anything
        # set by required/addition/retention. An empty dict is a valid setting
        # for retention therefore use None instead of empty dict.
        self.retention = None
        self.rename = {}
        return {}

    @abc.abstractmethod
    def process(self, inputs):
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

    def load_cache(self, filename=None):
        """
        Defines the behavior of how to load the cache file from the `filename`.
        Node can override this method. Default implementation assumes cudf
        dataframes.

        Arguments
        -------
        filename: str
            filename of the cache file. Leave as none to use default.

        """
        cache_dir = os.getenv('GQUANT_CACHE_DIR', self.cache_dir)
        if filename is None:
            filename = cache_dir + '/' + self.uid + '.hdf5'

        output_df = {}
        with pd.HDFStore(filename, mode='r') as hf:
            for oport, pspec in \
                    self._get_output_ports(full_port_spec=True).items():
                ptype = pspec.get(PortsSpecSchema.port_type)
                if self.outport_connected(oport):
                    ptype = ([ptype] if not isinstance(ptype,
                                                       list) else ptype)
                    key = '{}/{}'.format(self.uid, oport)
                    # check hdf store for the key
                    if key not in hf:
                        raise Exception(
                            'The task "{}" port "{}" key "{}" not found in'
                            'the hdf file "{}". Cannot load from cache.'
                            .format(self.uid, oport, key, filename)
                        )
                    if cudf.DataFrame not in ptype:
                        warnings.warn(
                            RuntimeWarning,
                            'Task "{}" port "{}" port type is not set to '
                            'cudf.DataFrame. Attempting to load port data '
                            'with cudf.read_hdf.'.format(self.uid, oport))
                    output_df[oport] = cudf.read_hdf(hf, key)
        return output_df

    def save_cache(self, output_df):
        '''Defines the behavior for how to save the output of a node to
        filesystem cache. Default implementation assumes cudf dataframes.

        :param output_df: The output from :meth:`process`. For saving to hdf
            requires that the dataframe(s) have `to_hdf` method.
        '''
        cache_dir = os.getenv('GQUANT_CACHE_DIR', self.cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        filename = cache_dir + '/' + self.uid + '.hdf5'
        with pd.HDFStore(filename, mode='w') as hf:
            for oport, odf in output_df.items():
                # check for to_hdf attribute
                if not hasattr(odf, 'to_hdf'):
                    raise Exception(
                        'Task "{}" port "{}" output object is missing '
                        '"to_hdf" attribute. Cannot save to cache.'
                        .format(self.uid, oport))

                dtype = '{}'.format(type(odf)).lower()
                if 'dataframe' not in dtype:
                    warnings.warn(
                        RuntimeWarning,
                        'Task "{}" port "{}" port type is not a dataframe.'
                        ' Attempting to save to hdf with "to_hdf" method.'
                        .format(self.uid, oport))
                key = '{}/{}'.format(self.uid, oport)
                odf.to_hdf(hf, key, format='table', data_columns=True)
