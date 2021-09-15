"""
 ////////////////////////////////////////////////////////////////////////////
 //
 // Copyright (C) NVIDIA Corporation.  All rights reserved.
 //
 // NVIDIA Sample Code
 //
 // Please refer to the NVIDIA end user license agreement (EULA) associated
 // with this source code for terms and conditions that govern your use of
 // this software. Any use, reproduction, disclosure, or distribution of
 // this software and related documentation outside the terms of the EULA
 // is strictly prohibited.
 //
 ////////////////////////////////////////////////////////////////////////////
"""

import cudf
from greenflow.dataframe_flow import Node, MetaData
from greenflow.dataframe_flow import NodePorts, PortsSpecSchema
from greenflow.dataframe_flow.util import get_file_path
from greenflow.dataframe_flow import ConfSchema


class LoadCsvNode(Node):

    def ports_setup(self):
        input_ports = {}
        output_ports = {
            'df_out': {
                PortsSpecSchema.port_type: cudf.DataFrame
            }
        }
        return NodePorts(inports=input_ports, outports=output_ports)

    def conf_schema(self):
        json = {
            "title": "Load stock data",
            "type": "object",
            "properties": {
                "csvfile":  {
                    "type": "string",
                    "description": "csv tick data"
                },
                "17assets":  {
                    "type": "boolean",
                    "description": "17 assets dataset"
                }
            },
            "required": ["csvfile"],
        }

        ui = {
            "csvfile": {"ui:widget": "CsvFileSelector"}
        }
        return ConfSchema(json=json, ui=ui)

    def init(self):
        pass

    def meta_setup(self):
        df_out_10 = {
            'date': 'date',
            'AAA': 'float64',
            'BBB': 'float64',
            'CCC': 'float64',
            'DDD': 'float64',
            'EEE': 'float64',
            'FFF': 'float64',
            'GGG': 'float64',
            'HHH': 'float64',
            'III': 'float64',
            'JJJ': 'float64',
        }

        df_out_17 = {
            'date': 'date',
            'BZA Index (Equities)': 'float64',
            'CLA Comdty (Commodities)': 'float64',
            'CNA Comdty (Fixed Income)': 'float64',
            'ESA Index (Equities)': 'float64',
            'G A Comdty (Fixed Income)': 'float64',
            'GCA Comdty (Commodities)': 'float64',
            'HIA Index (Equities)': 'float64',
            'NKA Index (Equities)': 'float64',
            'NQA Index (Equities)': 'float64',
            'RXA Comdty (Fixed Income)': 'float64',
            'SIA Comdty (Commodities)': 'float64',
            'SMA Index (Equities)': 'float64',
            'TYA Comdty (Fixed Income)': 'float64',
            'VGA Index (Equities)': 'float64',
            'XMA Comdty (Fixed Income)': 'float64',
            'XPA Index (Equities)': 'float64',
            'Z A Index (Equities)': 'float64',
        }
        assets_17 = self.conf.get('17assets', False)
        columns_out = {
        }
        columns_out['df_out'] = df_out_17 if assets_17 else df_out_10
        return MetaData(inports={}, outports=columns_out)

    def process(self, inputs):
        import dask.distributed
        try:
            client = dask.distributed.client.default_client()
        except ValueError:
            from dask_cuda import LocalCUDACluster
            cluster = LocalCUDACluster()
            from dask.distributed import Client
            client = Client(cluster)  # noqa
            print('start new Cluster')
        filename = get_file_path(self.conf['csvfile'])
        df = cudf.read_csv(filename, parse_dates=[0])
        df.columns = ['date']+[c for c in df.columns][1:]
        output = {}
        if self.outport_connected('df_out'):
            output.update({'df_out': df})
        return output
