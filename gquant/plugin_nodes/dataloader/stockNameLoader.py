from gquant.dataframe_flow import Node
import cudf
from gquant.dataframe_flow.portsSpecSchema import (PortsSpecSchema,
                                                   NodePorts,
                                                   ConfSchema)

STOCK_NAME_PORT_NAME = 'stock_name'


class StockNameLoader(Node):

    def ports_setup(self):
        input_ports = {}
        output_ports = {
            STOCK_NAME_PORT_NAME: {
                PortsSpecSchema.port_type: cudf.DataFrame
            }
        }
        return NodePorts(inports=input_ports, outports=output_ports)

    def init(self):
        self.required = {}

    def columns_setup(self):
        self.required = {}
        column_types = {"asset": "int64",
                        "asset_name": "object"}
        out_cols = {
            STOCK_NAME_PORT_NAME: column_types,
        }
        return out_cols

    def conf_schema(self):
        json = {
            "title": "Stock name csv file loader configure",
            "type": "object",
            "description": "Load the stock name data from the csv file",
            "properties": {
                "file":  {
                    "type": "string",
                    "description": "stock name csv file with full path"
                }
            },
            "required": ["file"],
        }

        ui = {
            "file": {"ui:widget": "text"},
        }
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        Load the csv file mapping stock id to symbol name into cudf DataFrame

        Arguments
        -------
         inputs: list
             empty list
        Returns
        -------
        cudf.DataFrame
        """

        name_df = cudf.read_csv(self.conf['file'])[['SM_ID', 'SYMBOL']]
        # change the names
        name_df.columns = ["asset", 'asset_name']
        return {STOCK_NAME_PORT_NAME: name_df}
