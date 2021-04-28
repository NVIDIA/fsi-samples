import re
import importlib
from .portsSpecSchema import (PortsSpecSchema, NodePorts)
from .metaSpec import (MetaDataSchema, MetaData)


__all__ = ['NodeExtensionMixin']

TYPES_CACHE = {}


class NodeExtensionMixin:
    def _sep_variable(self, variable):
        assert isinstance(variable, str)
        e = re.search('^\${(.*):(.*)}$', variable) # noqa
        if e is None and variable.startswith('$'):
            raise ValueError("varaible format is wrong")
        if e is None:
            return None
        groups = e.groups()
        return groups

    def _parse_variable(self, variable, port_inports):
        if isinstance(variable, int):
            return variable

        if isinstance(variable, dict):
            return variable

        if variable is None:
            return None

        port_type = PortsSpecSchema.port_type
        groups = self._sep_variable(variable)
        if groups is None:
            return variable

        if groups[0] == 'conf':
            return self.conf[groups[1]]
        elif groups[0] == 'port':
            return port_inports[groups[1]][port_type]
        else:
            raise KeyError('Cannot parse variable {}'.format(groups))

    def _load_type(self, type_str):
        return_list = False
        if isinstance(type_str, list):
            return_list = True
            type_str_list = type_str
        else:
            type_str_list = [type_str]

        clsobj_list = []
        for type_str in type_str_list:
            if type_str in TYPES_CACHE:
                clsobj_list.append(TYPES_CACHE[type_str])

            if isinstance(type_str, type):
                clsobj = type_str
            elif isinstance(type_str, str):
                splits = type_str.split('.')
                mod_str = ".".join(splits[:-1])
                mod = importlib.import_module(mod_str)
                clsobj = getattr(mod, splits[-1])
                TYPES_CACHE[type_str] = clsobj
            else:
                raise Exception('Cannot load type: {}'.format(type_str))

            clsobj_list.append(clsobj)

        if return_list:
            return clsobj_list
        else:
            return clsobj_list[0]

    def _resolve_ports(self, ports_template):
        '''The ports can be defined via template specification.

        Example:
            port_inports = {
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
                        PortsSpecSchema.DYN_MATCH: ['type0', 'type1']
                    }
                },
                ...
            }

            port_outports = {
                "port0_name": {
                    PortsSpecSchema.port_type: ["type0", "type1"]
                },
                "port1_name": {
                    PortsSpecSchema.port_type: "${port:port0_name}"
                },
                ...
            }


            ports_template = NodePorts(inports=port_inports,
                                       outports=port_outports)
            ports_resolved = self._resolve_ports(ports_template)

        Above, the types are specified as strings and loaded dynamically.
        Additionally an input port can use "dynamic" syntax for automatically
        resolving types for the input connections to that port. The output
        ports can similarlly define types as string to be loaded dynamically,
        and make references to port inputs to re-use an input port's types.
        After calling _resolve_ports the ports definitions would look something
        as follows:
            ports_resolved.inports == {
                "port0_name": {
                    PortsSpecSchema.port_type: [type0, type1]
                },
                "port1_name": {
                    PortsSpecSchema.port_type: "${conf:some_type}",
                    PortsSpecSchema.dynamic: {
                        PortsSpecSchema.DYN_MATCH: [type0, type1]
                    }
                },
                ...
            }

            ports_resolved.inports == {
                "port0_name": {
                    PortsSpecSchema.port_type: [type0, type1]
                },
                "port1_name": {
                    PortsSpecSchema.port_type: "${port:port0_name}"
                },
                ...
            }

        Port types using "$" syntax are resolved when the node is within a
        taskgraph context. This additional resolve logic is handled in
        :class:`NodeTaskGraphExtensionMixin.port_setup_ext`.

        :param ports_template: Ports definition via convenience templating.
        :type ports_template: NodePorts

        :returns: Resolved ports.
        :rtype: NodePorts

        '''
        ports = ports_template
        dy = PortsSpecSchema.dynamic
        port_type = PortsSpecSchema.port_type
        # resolve all the variables
        port_inports = {}
        inports = ports.inports
        for key in inports:
            key_name = self._parse_variable(key, inports)
            value = inports[key]
            ptype = value[port_type]
            return_list = False
            if isinstance(ptype, list):
                return_list = True
                ptype_list = ptype
            else:
                ptype_list = [ptype]

            loaded_types = [
                self._load_type(self._parse_variable(item, inports))
                if not isinstance(item, type) else item
                for item in ptype_list
            ]

            if return_list:
                value[port_type] = loaded_types
            else:
                value[port_type] = loaded_types[0]

            if dy in value:
                dynamic_value = value[dy]
                m_outputs = dynamic_value[PortsSpecSchema.DYN_MATCH]
                if isinstance(m_outputs, bool):
                    pass
                elif isinstance(m_outputs, list):
                    dynamic_value[PortsSpecSchema.DYN_MATCH] = [
                        self._load_type(self._parse_variable(item, inports))
                        if not isinstance(item, type) else item
                        for item in m_outputs
                    ]
                elif isinstance(m_outputs, str):
                    dynamic_value[PortsSpecSchema.DYN_MATCH] = self._load_type(
                        self._parse_variable(m_outputs, inports))
                else:
                    raise ValueError

            port_inports[key_name] = value

        # resolve all the variables
        port_outports = {}
        outports = ports.outports
        for key in outports:
            key_name = self._parse_variable(key, port_inports)
            value = outports[key]
            if isinstance(value[port_type], list):
                value[port_type] = [
                    self._load_type(self._parse_variable(item, port_inports))
                    if not isinstance(item, type) else item
                    for item in value[port_type]
                ]
            elif isinstance(value[port_type], str):
                # This part is valid if node is part of NodeTaskGraphMixin
                if not value[port_type].startswith('$'):
                    value[port_type] = self._load_type(
                        self._parse_variable(value[port_type],
                                             port_inports))
                else:
                    # it will be resolved inside the port_setup_ext
                    pass
            elif isinstance(value[port_type], type):
                pass
            else:
                raise ValueError

            port_outports[key_name] = value

        return NodePorts(inports=port_inports, outports=port_outports)

    def _resolve_meta(self, meta_template, port_inports):
        meta = meta_template
        meta_inports = {}
        metainports = meta.inports

        for key in metainports:
            key_name = self._parse_variable(key, port_inports)
            value = metainports[key]

            new_value = {}
            for vk in value:
                nvk = self._parse_variable(vk, port_inports)
                new_value[nvk] = self._parse_variable(value[vk],
                                                      port_inports)
            meta_inports[key_name] = new_value

        meta_outports = {}
        metaoutports = meta.outports
        data_accessor = MetaDataSchema.META_DATA
        order_accessor = MetaDataSchema.META_ORDER
        for key in metaoutports:
            meta_outports[key] = metaoutports[key].copy()

            key_name = self._parse_variable(key, port_inports)
            value = metaoutports[key]

            if data_accessor in value:
                new_data = {}
                for vk in value[data_accessor]:
                    nvk = self._parse_variable(vk, port_inports)
                    new_data[nvk] = self._parse_variable(
                        value[data_accessor][vk], port_inports)

                meta_outports[key_name][data_accessor] = new_data

            if order_accessor in value:
                new_order = {}
                for vk in value[order_accessor]:
                    nvk = self._parse_variable(vk, port_inports)
                    new_order[nvk] = value[order_accessor][vk]
                meta_outports[key_name][order_accessor] = new_order

        return MetaData(inports=meta_inports, outports=meta_outports)
