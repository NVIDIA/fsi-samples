from .portsSpecSchema import (PortsSpecSchema, NodePorts)
from .metaSpec import (MetaDataSchema, MetaData)
from copy import deepcopy

__all__ = ['NodeTaskGraphExtensionMixin']


class NodeTaskGraphExtensionMixin:
    '''Extension logic for a Node within a taskgraph. This mixin is used with
    NodeTaskGraphMixin.
    '''

    def reset_cache(self):
        '''Delete ivars that maintain the cache state for a node.'''
        if hasattr(self, 'ports_setup_cache'):
            del self.ports_setup_cache

        if hasattr(self, 'input_meta_cache'):
            del self.input_meta_cache

        if hasattr(self, 'input_connections_cache'):
            del self.input_connections_cache

        if hasattr(self, 'meta_data_cache'):
            del self.meta_data_cache

    def cache_update_result(self):
        self.reset_cache()
        # cache all the intermediate results
        self.ports_setup_cache = self.ports_setup()
        self.input_meta_cache = self.get_input_meta()
        self.input_connections_cache = self.get_connected_inports()
        self.meta_data_cache = self.meta_setup()

    def ports_setup_ext(self, ports):
        '''
        1. Finds the port type by the connected node port type.
        2. Set the port type to the determined type.
        3. If the node is not connected, it will use the list of the types.
        4. Also handles the dynamic port type.
        5. Set the port type to the determined type (from the graph topology)
           for the dynamic port.

        :param ports: These are resolved ports.
        :type ports: NodePorts

        :return: Node ports
        :rtype: NodePorts
        '''
        port_inports = ports.inports
        port_outports = ports.outports

        port_type = PortsSpecSchema.port_type
        dy = PortsSpecSchema.dynamic
        input_connections = self.get_connected_inports()

        dynamic = None
        inports = {}
        for input_port in port_inports:
            inports[input_port] = deepcopy(port_inports[input_port])
            if input_port in input_connections:
                determined_type = input_connections[input_port]
                inports[input_port].update({port_type: determined_type})

            if dy in port_inports[input_port]:
                inports[input_port][dy] = True
                dynamic = \
                    port_inports[input_port][dy][PortsSpecSchema.DYN_MATCH]

        outports = {}
        for output_port in port_outports:
            types = port_outports[output_port][port_type]
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
                outports[output_port] = {port_type: types}

        static_inport_names = [
            iport
            for iport in port_inports if not inports[iport].get(dy, False)
        ]

        for input_port in port_inports:
            dynamic = None
            if dy in port_inports[input_port]:
                dynamic = \
                    port_inports[input_port][dy][PortsSpecSchema.DYN_MATCH]

            if dynamic is not None:
                types = None
                if not isinstance(dynamic, bool):
                    types = dynamic

                for port_name in input_connections.keys():
                    # if port_name not in port_inports:
                    if port_name not in outports and \
                            port_name not in static_inport_names:
                        if types is not None:
                            outports[port_name] = {port_type: types}
                        else:
                            if isinstance(dynamic, bool) and dynamic:
                                types = input_connections[port_name]
                                outports[port_name] = {port_type: types}

        return NodePorts(inports=inports, outports=outports)

    def meta_setup_ext(self, meta):
        '''
        1. Based on meta operators, calculate the output meta
        2. Adjust meta data element orders based on specified order
        3. Pass the meta data for dynamically added output ports

        :param meta: the meta information that needs to be calculated.
        :type MetaData:  MetaData

        :return: MetaData
        :rtype: MetaData
        '''
        input_meta = self.get_input_meta()

        inports = meta.inports.copy()
        metaoutports = meta.outports
        outports = {}
        data_accessor = MetaDataSchema.META_DATA
        order_accessor = MetaDataSchema.META_ORDER
        for out_port_name in metaoutports:
            type_str = metaoutports[out_port_name].get(MetaDataSchema.META_OP)
            if type_str is None:
                # NOT A META_OP
                outports[out_port_name] = metaoutports[out_port_name]
            elif type_str == MetaDataSchema.META_OP_ADDITION:
                input_port = \
                    metaoutports[out_port_name][MetaDataSchema.META_REF_INPUT]
                if input_port in input_meta:
                    input_meta_data = input_meta[input_port]
                else:
                    input_meta_data = inports[input_port]

                outports[out_port_name] = input_meta_data.copy()
                outports[out_port_name].update(
                    metaoutports[out_port_name][data_accessor])
            elif type_str == MetaDataSchema.META_OP_RETENTION:
                outports[out_port_name] = \
                    metaoutports[out_port_name][data_accessor].copy()
            elif type_str == MetaDataSchema.META_OP_DELETION:
                input_port = metaoutports[out_port_name][
                    MetaDataSchema.META_REF_INPUT]
                if input_port in input_meta:
                    input_meta_data = input_meta[input_port]
                else:
                    input_meta_data = inports[input_port]

                outports[out_port_name] = input_meta_data.copy()
                for key in metaoutports[out_port_name][data_accessor]:
                    if key in outports[out_port_name]:
                        del outports[out_port_name][key]
            else:
                raise NotImplementedError('META_OP "{}" not implemented'
                                          .format(type_str))

            # adjust the columns order
            if order_accessor in metaoutports[out_port_name]:
                total_properties = len(outports[out_port_name])
                order_dict = metaoutports[out_port_name][order_accessor].copy()
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
        port_inports = self.ports_setup().inports
        static_inport_names = [
            iport
            for iport in port_inports if not port_inports[iport].get(dy, False)
        ]
        for input_port in port_inports:
            isdynamic = None
            if dy in port_inports[input_port]:
                isdynamic = port_inports[input_port][dy]

            if isdynamic is not None and isdynamic:
                input_connections = self.get_connected_inports()
                for port_name in input_connections.keys():
                    if port_name not in metaoutports and \
                            port_name not in static_inport_names and \
                            port_name in input_meta:
                        outports[port_name] = input_meta[port_name]

        return MetaData(inports=inports, outports=outports)
