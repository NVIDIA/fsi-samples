from gquant.dataframe_flow import Node
import cudf
from gquant.dataframe_flow.portsSpecSchema import (PortsSpecSchema,
                                                   NodePorts,
                                                   ConfSchema)
from gquant.dataframe_flow.cache import (cache_columns)
from gquant.dataframe_flow.util import get_file_path

from .stockMap import StockMap
STOCK_NAME_PORT_NAME = 'stock_name'
STOCK_MAP_PORT_NAME = 'map_data'


class StockNameLoader(Node):

    def _compute_hash_key(self):
        return hash((self.uid, self.conf['file']))

    def ports_setup(self):
        input_ports = {}
        output_ports = {
            STOCK_NAME_PORT_NAME: {
                PortsSpecSchema.port_type: cudf.DataFrame
            },
            STOCK_MAP_PORT_NAME: {
                PortsSpecSchema.port_type: StockMap
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
        if self.outport_connected(STOCK_MAP_PORT_NAME):
            if 'file' in self.conf:
                hash_key = self._compute_hash_key()
                if hash_key in cache_columns:
                    out_cols.update({
                        STOCK_MAP_PORT_NAME: cache_columns[hash_key]})
                else:
                    path = get_file_path(self.conf['file'])
                    name_df = cudf.read_csv(path)[['SM_ID', 'SYMBOL']]
                    name_df.columns = ["asset", 'asset_name']
                    pdf = name_df.to_pandas()
                    column_data = pdf.to_dict('list')
                    cache_columns[hash_key] = column_data
                    out_cols.update({STOCK_MAP_PORT_NAME: column_data})
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
            "file": {"ui:widget": "CsvFileSelector"}
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
        output = {}
        if self.outport_connected(STOCK_NAME_PORT_NAME):
            path = get_file_path(self.conf['file'])
            name_df = cudf.read_csv(path)[['SM_ID', 'SYMBOL']]
            # change the names
            name_df.columns = ["asset", 'asset_name']
            output.update({STOCK_NAME_PORT_NAME: name_df})
        if self.outport_connected(STOCK_MAP_PORT_NAME):
            output.update({STOCK_MAP_PORT_NAME: StockMap()})
        return output
