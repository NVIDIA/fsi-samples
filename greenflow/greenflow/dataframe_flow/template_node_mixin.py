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
                    PortsSpecSchema.DYN_MATCH: ['type0', 'type1']
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

        """
        if not hasattr(self, 'port_inports'):
            self.port_inports = {}

        if not hasattr(self, 'port_outports'):
            self.port_outports = {}

        if not hasattr(self, 'meta_inports'):
            self.meta_inports = {}

        if not hasattr(self, 'meta_outports'):
            self.meta_outports = {}

    def update(self):
        '''Updates state of a Node with resolved ports and meta.
        '''
        ports_template = \
            NodePorts(inports=self.port_inports, outports=self.port_outports)
        ports = self._resolve_ports(ports_template)
        port_inports = ports.inports
        meta_template = \
            MetaData(inports=self.meta_inports, outports=self.meta_outports)
        meta = self._resolve_meta(meta_template, port_inports)

        self.port_inports = ports.inports
        self.port_outports = ports.outports

        self.meta_inports = meta.inports
        self.meta_outports = meta.outports

    def ports_setup(self):
        ports = NodePorts(inports=self.port_inports,
                          outports=self.port_outports)
        return self.ports_setup_ext(ports)

    def meta_setup(self):
        meta = MetaData(inports=self.meta_inports, outports=self.meta_outports)
        return self.meta_setup_ext(meta)
