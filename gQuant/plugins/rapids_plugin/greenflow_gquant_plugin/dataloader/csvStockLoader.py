from greenflow.dataframe_flow import Node
from greenflow.dataframe_flow.portsSpecSchema import (PortsSpecSchema,
                                                   NodePorts,
                                                   MetaData,
                                                   ConfSchema)
import cudf
import dask_cudf
import pandas as pd
from greenflow.dataframe_flow.util import get_file_path
from .._port_type_node import _PortTypesMixin

CUDF_PORT_NAME = 'cudf_out'
DASK_CUDF_PORT_NAME = 'dask_cudf_out'
PANDAS_PORT_NAME = 'pandas_out'


class CsvStockLoader(_PortTypesMixin, Node):

    def ports_setup(self):
        input_ports = {}
        output_ports = {
            CUDF_PORT_NAME: {
                PortsSpecSchema.port_type: cudf.DataFrame
            },
            DASK_CUDF_PORT_NAME: {
                PortsSpecSchema.port_type: dask_cudf.DataFrame
            },
            PANDAS_PORT_NAME: {
                PortsSpecSchema.port_type: pd.DataFrame
            }
        }
        return NodePorts(inports=input_ports, outports=output_ports)

    def init(self):
        pass

    def meta_setup(self):
        column_types = {"datetime": "date",
                        "asset": "int64",
                        "volume": "float64",
                        "close": "float64",
                        "open": "float64",
                        "high": "float64",
                        "low": "float64"}
        out_cols = {
            CUDF_PORT_NAME: column_types,
            DASK_CUDF_PORT_NAME: column_types,
            PANDAS_PORT_NAME: column_types
        }
        required = {}
        metadata = MetaData(inports=required, outports=out_cols)
        return metadata

    def conf_schema(self):
        json = {
            "title": "Stock csv data loader configure",
            "type": "object",
            "description": "Load the stock daily bar data from the csv file",
            "properties": {
                "file":  {
                    "type": "string",
                    "description": "stock csv data file with full path"
                },
                "path":  {
                    "type": "string",
                    "description": "path to the directory for csv files"
                }
            }
        }

        ui = {
            "file": {"ui:widget": "CsvFileSelector"},
            "path": {"ui:widget": "PathSelector"}
        }
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        Load the end of day stock CSV data into cuDF dataframe

        Arguments
        -------
         inputs: list
             empty list
        Returns
        -------
        cudf.DataFrame
        """
        output = {}
        if self.outport_connected(CUDF_PORT_NAME):
            path = get_file_path(self.conf['file'])
            df = cudf.read_csv(path)
            # extract the year, month, day
            ymd = df['DTE'].astype(
                'str').str.extract(r'(\d\d\d\d)(\d\d)(\d\d)')
            # construct the standard datetime str
            df['DTE'] = ymd[0].str.cat(
                ymd[1],
                '-').str.cat(ymd[2], '-').astype('datetime64[ms]')
            df = df[['DTE', 'OPEN', 'CLOSE', 'HIGH', 'LOW', 'SM_ID', 'VOLUME']]
            df['VOLUME'] /= 1000
            # change the names
            df.columns = ['datetime', 'open', 'close',
                          'high', 'low', "asset", 'volume']
            output.update({CUDF_PORT_NAME: df})
        if self.outport_connected(PANDAS_PORT_NAME):
            path = get_file_path(self.conf['file'])
            df = pd.read_csv(path,
                             converters={'DTE':
                                         lambda x: pd.Timestamp(str(x))})
            df = df[['DTE', 'OPEN',
                     'CLOSE', 'HIGH',
                     'LOW', 'SM_ID', 'VOLUME']]
            df['VOLUME'] /= 1000
            df.columns = ['datetime', 'open', 'close', 'high',
                          'low', "asset", 'volume']
            output.update({PANDAS_PORT_NAME: df})
        if self.outport_connected(DASK_CUDF_PORT_NAME):
            path = get_file_path(self.conf['path'])
            df = dask_cudf.read_csv(path+'/*.csv',
                                    parse_dates=['datetime'])
            output.update({DASK_CUDF_PORT_NAME: df})
        return output
