from gquant.dataframe_flow import Node
from gquant.dataframe_flow._port_type_node import _PortTypesMixin
from gquant.dataframe_flow.portsSpecSchema import (ConfSchema,
                                                   PortsSpecSchema, NodePorts)
from ..dataloader.stockMap import StockMap


class AssetFilterNode(Node, _PortTypesMixin):

    def init(self):
        _PortTypesMixin.init(self)
        self.INPUT_PORT_NAME = 'stock_in'
        self.OUTPUT_PORT_NAME = 'stock_out'
        self.INPUT_MAP_NAME = 'name_map'
        cols_required = {"asset": "int64"}
        self.required = {
            self.INPUT_PORT_NAME: cols_required
        }

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
            }
        }

        input_connections = self.get_connected_inports()
        if self.INPUT_PORT_NAME in input_connections:
            determined_type = input_connections[self.INPUT_PORT_NAME]
            input_ports.update({self.INPUT_PORT_NAME:
                                {port_type: determined_type}})
            # connected
            return NodePorts(inports=input_ports,
                             outports={self.OUTPUT_PORT_NAME: {
                                 port_type: determined_type}})
        else:
            return NodePorts(inports=input_ports, outports=output_ports)

    def columns_setup(self):
        return _PortTypesMixin.columns_setup(self)

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
        input_columns = self.get_input_columns()
        if self.INPUT_MAP_NAME in input_columns:
            col_from_inport = input_columns[self.INPUT_MAP_NAME]
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
        return {self.OUTPUT_PORT_NAME: output_df}
