from greenflow.dataframe_flow import Node
from .._port_type_node import _PortTypesMixin
from greenflow.dataframe_flow.portsSpecSchema import (ConfSchema,
                                                   MetaData,
                                                   PortsSpecSchema, NodePorts)
from ..dataloader.stockMap import StockMap


class AssetFilterNode(_PortTypesMixin, Node):

    def init(self):
        _PortTypesMixin.init(self)
        self.INPUT_PORT_NAME = 'stock_in'
        self.OUTPUT_PORT_NAME = 'stock_out'
        self.INPUT_MAP_NAME = 'name_map'
        self.OUTPUT_ASSET_NAME = 'stock_name'

    def ports_setup_from_types(self, types):
        port_type = PortsSpecSchema.port_type
        input_ports = {
            self.INPUT_PORT_NAME: {
                port_type: types
            },
            self.INPUT_MAP_NAME: {
                port_type: StockMap
            }
        }

        output_ports = {
            self.OUTPUT_PORT_NAME: {
                port_type: types
            },
            self.OUTPUT_ASSET_NAME: {
                port_type: str
            }
        }

        input_connections = self.get_connected_inports()
        if self.INPUT_PORT_NAME in input_connections:
            determined_type = input_connections[self.INPUT_PORT_NAME]
            input_ports.update({self.INPUT_PORT_NAME:
                                {port_type: determined_type}})
            output_ports.update({self.OUTPUT_PORT_NAME: {
                                 port_type: determined_type}})
            # connected
            return NodePorts(inports=input_ports,
                             outports=output_ports)
        else:
            return NodePorts(inports=input_ports, outports=output_ports)

    def _find_asset_name(self):
        name = ""
        input_meta = self.get_input_meta()
        if self.outport_connected(self.OUTPUT_ASSET_NAME):
            if self.INPUT_MAP_NAME in input_meta and 'asset' in self.conf:
                col_from_inport = input_meta[self.INPUT_MAP_NAME]
                enums = col_from_inport['asset']
                enumNames = col_from_inport['asset_name']
                found = False
                for i, name in zip(enums, enumNames):
                    if i == self.conf['asset']:
                        found = True
                        break
                if not found:
                    name = ""
        return name

    def meta_setup(self):
        cols_required = {"asset": "int64"}
        required = {
            self.INPUT_PORT_NAME: cols_required
        }
        input_meta = self.get_input_meta()
        name = self._find_asset_name()
        if self.INPUT_PORT_NAME in input_meta:
            col_from_inport = input_meta[self.INPUT_PORT_NAME]
            output_cols = {
                self.OUTPUT_PORT_NAME: col_from_inport,
                self.OUTPUT_ASSET_NAME: {"asset_name": name}
            }
            metadata = MetaData(inports=required, outports=output_cols)
            return metadata
        else:
            col_from_inport = required[self.INPUT_PORT_NAME]
            output_cols = {
                self.OUTPUT_PORT_NAME: col_from_inport,
                self.OUTPUT_ASSET_NAME: {"asset_name": name}
            }
            metadata = MetaData(inports=required, outports=output_cols)
            return metadata

    def ports_setup(self):
        return _PortTypesMixin.ports_setup(self)

    def conf_schema(self):
        json = {
            "title": "Asset Filter Node configure",
            "type": "object",
            "description": "select the asset based on asset id",
            "properties": {
                "asset":  {
                    "type": "number",
                    "description": "asset id number"
                }
            },
            "required": ["asset"],
        }
        ui = {
        }
        input_meta = self.get_input_meta()
        if self.INPUT_MAP_NAME in input_meta:
            col_from_inport = input_meta[self.INPUT_MAP_NAME]
            enums = col_from_inport['asset']
            enumNames = col_from_inport['asset_name']
            json['properties']['asset']['enum'] = enums
            json['properties']['asset']['enumNames'] = enumNames
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        select the asset based on asset id, which is defined in `asset` in the
        nodes' conf

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        input_df = inputs[self.INPUT_PORT_NAME]
        output_df = input_df.query('asset==%s' % self.conf["asset"])
        output = {self.OUTPUT_PORT_NAME: output_df}
        if self.outport_connected(self.OUTPUT_ASSET_NAME):
            name = self._find_asset_name()
            output.update({self.OUTPUT_ASSET_NAME: name})
        return output
