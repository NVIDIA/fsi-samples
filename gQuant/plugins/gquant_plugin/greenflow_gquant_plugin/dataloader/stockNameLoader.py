from greenflow.dataframe_flow import Node
import cudf
from greenflow.dataframe_flow.portsSpecSchema import (
    PortsSpecSchema, NodePorts, ConfSchema)
from greenflow.dataframe_flow.metaSpec import MetaData
from ..cache import CACHE_NAME
from greenflow.dataframe_flow.util import get_file_path

from .stockMap import StockMap
from ..node_hdf_cache import NodeHDFCacheMixin

STOCK_NAME_PORT_NAME = 'stock_name'
STOCK_MAP_PORT_NAME = 'map_data'


class StockNameLoader(NodeHDFCacheMixin, Node):

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

    def meta_setup(self):
        required = {}
        column_types = {"asset": "int64",
                        "asset_name": "object"}
        out_cols = {
            STOCK_NAME_PORT_NAME: column_types,
        }
        if self.outport_connected(STOCK_MAP_PORT_NAME):
            if 'file' in self.conf:
                hash_key = self._compute_hash_key()
                if hash_key in CACHE_NAME:
                    out_cols.update({
                        STOCK_MAP_PORT_NAME: CACHE_NAME[hash_key]})
                else:
                    path = get_file_path(self.conf['file'])
                    name_df = cudf.read_csv(path)[['SM_ID', 'SYMBOL']]
                    name_df.columns = ["asset", 'asset_name']
                    pdf = name_df.to_pandas()
                    column_data = pdf.to_dict('list')
                    CACHE_NAME[hash_key] = column_data
                    out_cols.update({STOCK_MAP_PORT_NAME: column_data})
        metadata = MetaData(inports=required, outports=out_cols)
        return metadata

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
