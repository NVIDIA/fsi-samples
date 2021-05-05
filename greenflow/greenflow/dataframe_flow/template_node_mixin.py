from greenflow.dataframe_flow.portsSpecSchema import NodePorts
from greenflow.dataframe_flow.metaSpec import MetaData


__all__ = ['TemplateNodeMixin']


class TemplateNodeMixin:
    '''This mixin is used with Nodes that use attributes for managing ports
    and meta.

    :ivar port_inports: Ports dictionary for input ports.
    :ivar port_outports: Ports dictionary for output ports.
    :ivar meta_inports: Metadata dictionary for input ports.
    :ivar meta_outports: Metadata dictionary for output ports.
    '''

    def init(self):
        """
        Used to initilze the Node. called from the node constructor
        all children should run parent init first in the constructor e.g.
        def init(self):
            TemplateNodeMixin.init(self)
            ....

        In this function. Define the static ports and meta setup. Note,
        only static information can be used includig the self.conf
        information. If need information from
        self.get_connected_inports() and self.get_input_meta(),
        please define it in update() function.

        Define the template ports setup by self.template_ports_setup
        E.g.

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
        self.template_ports_setup(in_ports=port_inports, 
                                  out_ports=port_outports)

        Define the template meta data setup by self.template_meta_setup.
        E.g.

        meta_inports = {
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
        meta_outports = {
            "port0_name": {
                MetaDataSchema.META_REF_INPUT: "port0_name",
                MetaDataSchema.META_OP: MetaDataSchema.META_OP_ADDITION,
                MetaDataSchema.META_DATA: {
                    "${conf:abc}": "type0",
                    "name1": "type1",
                    "name2": "${conf:type1}",
                }
                # order is optional
                MetaDataSchema.META_ORDER: {
                    "${conf:abc}": 0,
                    "name2": -1
                }
            },
            "port1_name": {
                MetaDataSchema.META_OP: MetaDataSchema.META_OP_RETENTION,
                MetaDataSchema.META_DATA: {
                    "${conf:abc}": "type0",
                    "name1": "type1",
                    "name2": "${conf:type1}",
                },
                # order is optional
                MetaDataSchema.META_ORDER: {
                    "${conf:abc}": -1,
                }
            },
            ...
        }
        self.template_meta_setup(in_ports=meta_inports,
                                 out_ports=meta_outports)
        """
        if not hasattr(self, '__port_inports'):
            self.__port_inports = {}

        if not hasattr(self, '__port_outports'):
            self.__port_outports = {}

        if not hasattr(self, '__meta_inports'):
            self.__meta_inports = {}

        if not hasattr(self, '__meta_outports'):
            self.__meta_outports = {}

    def template_ports_setup(self, in_ports=None, out_ports=None):
        if in_ports is not None:
            self.__port_inports = in_ports
        if out_ports is not None:
            self.__port_outports = out_ports
        return NodePorts(inports=self.__port_inports,
                         outports=self.__port_outports)

    def template_meta_setup(self, in_ports=None, out_ports=None):
        if in_ports is not None:
            self.__meta_inports = in_ports
        if out_ports is not None:
            self.__meta_outports = out_ports
        return MetaData(inports=self.__meta_inports,
                        outports=self.__meta_outports)

    def update(self):
        '''Updates state of a Node with resolved ports and meta.
        '''
        ports_template = \
            NodePorts(inports=self.__port_inports,
                      outports=self.__port_outports)
        ports = self._resolve_ports(ports_template)
        port_inports = ports.inports
        meta_template = \
            MetaData(inports=self.__meta_inports,
                     outports=self.__meta_outports)
        meta = self._resolve_meta(meta_template, port_inports)

        self.__port_inports = ports.inports
        self.__port_outports = ports.outports

        self.__meta_inports = meta.inports
        self.__meta_outports = meta.outports

    def ports_setup(self):
        ports = NodePorts(inports=self.__port_inports,
                          outports=self.__port_outports)
        return self.ports_setup_ext(ports)

    def meta_setup(self):
        meta = MetaData(inports=self.__meta_inports,
                        outports=self.__meta_outports)
        return self.meta_setup_ext(meta)
