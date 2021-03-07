from greenflow.dataframe_flow.portsSpecSchema import (PortsSpecSchema,
                                                      MetaData, NodePorts)
import cudf
import os
import warnings
import pandas as pd
import re
import importlib


class SimpleNodeMixin(object):

    def init(self):
        """
        self.port_inports = {
            "port0_name": {
                PortsSpecSchema.port_type: ["type0", "type1"]
            },
            "port1_name": {
                PortsSpecSchema.port_type: "${conf:some_type}",
                PortsSpecSchema.dynamic: {
                    # choie can be True/False, list of types or string
                    # True, generate outports matching the
                    # connected dynamic input ports, use the
                    # the same type as the dynamic port
                    # False, not generate matching outports
                    # list of types or string, same as True condition,
                    # but use the specified types
                    self.DYN_MATCH: ['type0', 'type1']
                }
            },
            ...
        }
        self.port_outports = {
            "port0_name": {
                PortsSpecSchema.port_type: ["type0", "type1"]
            },
            "port1_name": {
                PortsSpecSchema.port_type: "${port:port0_name}"
            },
            ...
        }
        self.meta_inports = {
            "port0_name": {
                "name0": "type0",
                "name1": "type1",
                "name2": "type2",
            },
            "port1_name": {
                "${conf:abc}": "type0",
                "name1": "type1",
                "name2": "${conf:type1}",
            },
            ...
        }
        self.meta_outports = {
            "port0_name": {
                self.META_REF_INPUT:: "port0_name",
                self.META_OP: self.META_OP_ADDITION,
                self.META_DATA: {
                    "${conf:abc}": "type0",
                    "name1": "type1",
                    "name2": "${conf:type1}",
                }
                # order is optional
                self.META_ORDER: {
                    "${conf:abc}": 0,
                    "name2": -1
                }
            },
            "port1_name": {
                self.META_OP: self.META_OP_RETENTION,
                self.META_DATA: {
                    "${conf:abc}": "type0",
                    "name1": "type1",
                    "name2": "${conf:type1}",
                },
                # order is optional
                self.META_ORDER: {
                    "${conf:abc}": -1,
                }
            },
            ...
        }

        """
        self.port_inports = {}
        self.port_outports = {}
        self.meta_inports = {}
        self.meta_outports = {}
        self.META_OP_DELETION = 'deletion'
        self.META_OP_ADDITION = 'addition'
        self.META_OP_RETENTION = 'retention'
        self.META_REF_INPUT = 'input'
        self.META_OP = 'meta_op'
        self.META_DATA = 'data'
        self.META_ORDER = 'order'
        self.DYN_MATCH = 'matching_outputs'

    def update(self):
        dy = PortsSpecSchema.dynamic
        port_type = PortsSpecSchema.port_type
        # resolve all the variables
        port_inports = {}
        for key in self.port_inports:
            key_name = self._parse_variable(key)
            value = self.port_inports[key]
            if isinstance(value[port_type], list):
                value[port_type] = [
                    self._parse_variable(item) for item in value[port_type]
                ]
            elif isinstance(value[port_type], str):
                value[port_type] = self._parse_variable(value[port_type])
            else:
                raise ValueError
            if dy in value:
                dynamic_value = value[dy]
                m_outputs = dynamic_value[self.DYN_MATCH]
                if isinstance(m_outputs, bool):
                    pass
                elif isinstance(m_outputs, list):
                    dynamic_value[self.DYN_MATCH] = [
                        self._parse_variable(item) for item in m_outputs
                    ]
                elif isinstance(m_outputs, str):
                    dynamic_value[self.DYN_MATCH] = self._parse_variable(
                        m_outputs)
                else:
                    raise ValueError
            port_inports[key_name] = value
        self.port_inports = port_inports

        # resolve all the variables
        port_outports = {}
        for key in self.port_outports:
            key_name = self._parse_variable(key)
            value = self.port_outports[key]
            if isinstance(value[port_type], list):
                value[port_type] = [
                    self._parse_variable(item) for item in value[port_type]
                ]
            elif isinstance(value[port_type], str):
                # it will be resolved inside the port_setup
                pass
            else:
                raise ValueError
            port_outports[key_name] = value
        self.port_outports = port_outports

        meta_inports = {}
        for key in self.meta_inports:
            key_name = self._parse_variable(key)
            value = self.meta_inports[key]

            new_value = {}
            for vk in value:
                nvk = self._parse_variable(vk)
                new_value[nvk] = self._parse_variable(value[vk])
            meta_inports[key_name] = new_value
        self.meta_inports = meta_inports

        meta_outports = {}
        for key in self.meta_outports:
            meta_outports[key] = self.meta_outports[key].copy()

            key_name = self._parse_variable(key)
            value = self.meta_outports[key]

            new_data = {}
            for vk in value['data']:
                nvk = self._parse_variable(vk)
                new_data[nvk] = self._parse_variable(value['data'][vk])
            meta_outports[key_name]['data'] = new_data

            if 'order' in value:
                new_order = {}
                for vk in value['order']:
                    nvk = self._parse_variable(vk)
                    new_order[nvk] = value['order'][vk]
                meta_outports[key_name]['order'] = new_order
        self.meta_outports = meta_outports

    def _parse_variable(self, variable):
        if variable is None:
            return None
        port_type = PortsSpecSchema.port_type
        groups = self._sep_variable(variable)
        if groups is None:
            return variable
        if groups[0] == 'conf':
            return self.conf[groups[1]]
        elif groups[1] == 'port':
            return self.port_inports[groups[1]][port_type]
        else:
            raise KeyError('not recognized envirionable')

    def _sep_variable(self, variable):
        assert isinstance(variable, str)
        e = re.search('^\${(.*):(.*)}$', variable) # noqa
        if e is None and variable.startswith('$'):
            raise ValueError("varaible format is wrong")
        if e is None:
            return None
        groups = e.groups()
        return groups

    def ports_setup(self):
        port_type = PortsSpecSchema.port_type
        dy = PortsSpecSchema.dynamic
        input_connections = self.get_connected_inports()
        dynamic = None
        inports = {}
        for input_port in self.port_inports:
            if input_port in input_connections:
                determined_type = input_connections[input_port]
                inports[input_port] = {
                    port_type: determined_type
                }
            else:
                types = self.port_inports[input_port][port_type]
                # load the str type
                if isinstance(types, list):
                    types = [self._load_type(t) for t in types]
                elif isinstance(types, str):
                    types = self._load_type(types)
                else:
                    raise ValueError("not recongized type {}".format(types))
                inports[input_port] = {
                    port_type: types
                }
            if dy in self.port_inports[input_port]:
                inports[input_port][dy] = True
                dynamic = self.port_inports[input_port][dy][self.DYN_MATCH]
        outports = {}
        for output_port in self.port_outports:
            types = self.port_outports[output_port][port_type]
            if isinstance(types, str) and types.startswith('$'):
                groups = self._sep_variable(types)
                if groups[0] != 'port':
                    raise ValueError("expect variable {} refer to port".format(
                        groups[0]))
                if groups[1] not in inports:
                    raise ValueError(
                        "expect variable name {} refer to a inport name"
                        .format(groups[1]))
                input_port_name = groups[1]
                outports[output_port] = {
                    port_type:
                    inports[input_port_name][port_type]
                }
            else:
                # load the str type
                if isinstance(types, list):
                    types = [self._load_type(t) for t in types]
                elif isinstance(types, str):
                    types = self._load_type(types)
                else:
                    raise ValueError("not recongized type {}".format(types))
                outports[output_port] = {
                    port_type: types
                }
        if dynamic is not None:
            types = None
            if isinstance(dynamic, str):
                types = self._load_type(dynamic)
            elif isinstance(dynamic, list):
                types = [self._load_type(t) for t in dynamic]
            for port_name in input_connections.keys():
                if port_name not in self.port_inports:
                    if types is not None:
                        outports[port_name] = {port_type: types}
                    else:
                        if isinstance(dynamic, bool) and dynamic:
                            types = input_connections[port_name]
                            outports[port_name] = {port_type: types}
        return NodePorts(inports=inports, outports=outports)

    def _load_type(self, type_str):
        splits = type_str.split('.')
        mod_str = ".".join(splits[:-1])
        mod = importlib.import_module(mod_str)
        return getattr(mod, splits[-1])

    def meta_setup(self, required={}):
        input_meta = self.get_input_meta()
        inports = self.meta_inports.copy()
        outports = {}
        for out_port_name in self.meta_outports:
            type_str = self.meta_outports[out_port_name][self.META_OP]
            if type_str == self.META_OP_ADDITION:
                input_port = self.meta_outports[out_port_name][
                    self.META_REF_INPUT]
                if input_port in input_meta:
                    input_meta_data = input_meta[input_port]
                else:
                    input_meta_data = inports[input_port]
                outports[out_port_name] = input_meta_data.copy()
                outports[out_port_name].update(
                    self.meta_outports[out_port_name]['data'])
            elif type_str == self.META_OP_RETENTION:
                outports[out_port_name] = self.meta_outports[out_port_name][
                    'data'].copy()
            elif type_str == self.META_OP_DELETION:
                input_port = self.meta_outports[out_port_name][
                    self.META_REF_INPUT]
                if input_port in input_meta:
                    input_meta_data = input_meta[input_port]
                else:
                    input_meta_data = inports[input_port]
                outports[out_port_name] = input_meta_data.copy()
                for key in self.meta_outports[out_port_name]['data']:
                    if key in outports[out_port_name]:
                        del outports[out_port_name][key]
            else:
                raise NotImplementedError("not implmented {}".format(type_str))
            # adjust the columns order
            if 'order' in self.meta_outports[out_port_name]:
                total_properties = len(outports[out_port_name])
                order_dict = self.meta_outports[out_port_name]['order'].copy()
                key_lists = list(outports[out_port_name].keys())
                for key in order_dict:
                    if order_dict[key] < 0:
                        order_dict[key] += total_properties
                    if key in key_lists:
                        key_lists.remove(key)
                items = list(order_dict.items())
                items.sort(key=lambda x: x[1])
                for i in items:
                    key_lists.insert(i[1], i[0])
                old_dict = outports[out_port_name]
                outports[out_port_name] = {
                    k: old_dict[k]
                    for k in key_lists if k in old_dict
                }
        # handle the dynamic output meta_setup
        dy = PortsSpecSchema.dynamic
        dynamic = None
        for input_port in self.port_inports:
            if dy in self.port_inports[input_port]:
                dynamic = self.port_inports[input_port][dy][self.DYN_MATCH]
        if dynamic is not None:
            output_meta = False
            if isinstance(dynamic, str):
                output_meta = True
            elif isinstance(dynamic, list):
                output_meta = True
            elif isinstance(dynamic, bool) and dynamic:
                output_meta = True
            if output_meta:
                input_connections = self.get_connected_inports()
                for port_name in input_connections.keys():
                    if port_name not in self.port_inports:
                        outports[port_name] = input_meta[port_name]
        return MetaData(inports=inports, outports=outports)

    def load_cache(self, filename=None) -> dict:
        """
        Defines the behavior of how to load the cache file from the `filename`.
        Node can override this method. Default implementation assumes cudf
        dataframes.

        Arguments
        -------
        filename: str
            filename of the cache file. Leave as none to use default.
        returns: dict
            dictionary of the output from this node
        """
        cache_dir = os.getenv('GREENFLOW_CACHE_DIR', self.cache_dir)
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

    def save_cache(self, output_data: dict):
        '''Defines the behavior for how to save the output of a node to
        filesystem cache. Default implementation assumes cudf dataframes.

        :param output_data: The output from :meth:`process`. For saving to hdf
            requires that the dataframe(s) have `to_hdf` method.
        '''
        cache_dir = os.getenv('GREENFLOW_CACHE_DIR', self.cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        filename = cache_dir + '/' + self.uid + '.hdf5'
        with pd.HDFStore(filename, mode='w') as hf:
            for oport, odf in output_data.items():
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
