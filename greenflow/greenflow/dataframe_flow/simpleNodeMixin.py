from .portsSpecSchema import (PortsSpecSchema, MetaData, NodePorts)
import re
import importlib
module_cache = {}


class SimpleNodeMixin(object):

    def init(self):
        """
        Used to initilze the Node. called from the node constructore
        all children should run parent init first in the construtor e.g.
        def init(self):
            SimpleNodeMixin(self)
            ....

        In this function. Define the static ports and meta setup. Note,
        only static information can be used includig the self.conf
        information. If need information from
        self.get_connected_inports() and self.get_input_meta(),
        please define it in update() function.

        Define the ports setup in self.port_inports and self.port_outputs
        E.g.

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

        Define the meta data setup in self.meta_inports and
        self.meta_outports.
        E.g.

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

    def cache_update_result(self):
        # cache all the intermediate results
        self.ports_setup_cache = self.ports_setup()
        self.input_meta = self.get_input_meta()
        self.input_connections = self.get_connected_inports()
        self.meta_data_cache = self.meta_setup()

    def update(self):
        """
        This function is called after the computation graph is
        constructed by Taskgraph.build method. It assumes the
        graph structure is up to date

        If need information from self.get_connected_inports()
        or self.get_input_meta() to modify the ports_setup or
        meta_setup, this is the place to do it.

        Other dynamic information can be defined here too.
        For customized node, following is the pattern

        def update(self):
            SimpleNodeMixin.update(self)
            # your code to use latest graph structure
            # to update self.meta_inports, self.meta_outports.
            # Note, currently ports_setup is cached for the
            # first time calling it. Make sure all the information
            # for self.port_inports and self.port_outports
            # is properly set before calling self.get_connected_inports()
            # or self.get_input_meta() as they are calling ports_setup
            # inside
        """
        dy = PortsSpecSchema.dynamic
        port_type = PortsSpecSchema.port_type
        # resolve all the variables
        port_inports = {}
        if hasattr(self, 'port_inports'):
            for key in self.port_inports:
                key_name = self._parse_variable(key)
                value = self.port_inports[key]
                if isinstance(value[port_type], list):
                    value[port_type] = [
                        self._load_type(self._parse_variable(item))
                        for item in value[port_type]
                    ]
                elif isinstance(value[port_type], str):
                    value[port_type] = self._load_type(
                        self._parse_variable(value[port_type]))
                else:
                    raise ValueError
                if dy in value:
                    dynamic_value = value[dy]
                    m_outputs = dynamic_value[self.DYN_MATCH]
                    if isinstance(m_outputs, bool):
                        pass
                    elif isinstance(m_outputs, list):
                        dynamic_value[self.DYN_MATCH] = [
                            self._load_type(self._parse_variable(item))
                            for item in m_outputs
                        ]
                    elif isinstance(m_outputs, str):
                        dynamic_value[self.DYN_MATCH] = self._load_type(
                            self._parse_variable(m_outputs))
                    else:
                        raise ValueError
                port_inports[key_name] = value
            self.port_inports = port_inports

        # resolve all the variables
        port_outports = {}
        if hasattr(self, 'port_outports'):
            for key in self.port_outports:
                key_name = self._parse_variable(key)
                value = self.port_outports[key]
                if isinstance(value[port_type], list):
                    value[port_type] = [
                        self._load_type(self._parse_variable(item))
                        for item in value[port_type]
                    ]
                elif isinstance(value[port_type], str):
                    if not value[port_type].startswith('$'):
                        value[port_type] = self._load_type(
                            self._parse_variable(value[port_type]))
                    else:
                        # it will be resolved inside the port_setup
                        pass
                else:
                    raise ValueError
                port_outports[key_name] = value
            self.port_outports = port_outports

        meta_inports = {}
        if hasattr(self, 'meta_inports'):
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
        if hasattr(self, 'meta_outports'):
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
        if isinstance(variable, int):
            return variable
        elif isinstance(variable, dict):
            return variable
        elif variable is None:
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
        if hasattr(self, 'ports_setup_cache'):
            return self.ports_setup_cache
        port_type = PortsSpecSchema.port_type
        dy = PortsSpecSchema.dynamic
        if hasattr(self, 'input_connections'):
            input_connections = self.input_connections
        else:
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
                # if isinstance(types, list):
                #     types = [self._load_type(t) for t in types]
                # elif isinstance(types, str):
                #     types = self._load_type(types)
                # else:
                #     raise ValueError("not recongized type {}".format(types))
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
                # if isinstance(types, list):
                #     types = [self._load_type(t) for t in types]
                # elif isinstance(types, str):
                #     types = self._load_type(types)
                # else:
                #     raise ValueError("not recongized type {}".format(types))
                outports[output_port] = {
                    port_type: types
                }
        if dynamic is not None:
            types = None
            if not isinstance(dynamic, bool):
                types = dynamic
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
        if type_str in module_cache:
            return module_cache[type_str]
        splits = type_str.split('.')
        mod_str = ".".join(splits[:-1])
        mod = importlib.import_module(mod_str)
        clsobj = getattr(mod, splits[-1])
        module_cache[type_str] = clsobj
        return clsobj

    def meta_setup(self, required={}):
        if hasattr(self, 'meta_data_cache'):
            return self.meta_data_cache
        if hasattr(self, 'input_meta'):
            input_meta = self.input_meta
        else:
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
                if hasattr(self, 'input_connections'):
                    input_connections = self.input_connections
                else:
                    input_connections = self.get_connected_inports()
                for port_name in input_connections.keys():
                    if port_name not in self.port_inports:
                        outports[port_name] = input_meta[port_name]
        return MetaData(inports=inports, outports=outports)
