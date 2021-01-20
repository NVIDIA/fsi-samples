import numpy as np
import pandas as pd
from gquant.dataframe_flow import Node, MetaData
from gquant.dataframe_flow import NodePorts, PortsSpecSchema
from gquant.dataframe_flow import ConfSchema


class PointNode(Node):

    def ports_setup(self):
        input_ports = {}
        output_ports = {
            'points_df_out': {
                PortsSpecSchema.port_type: pd.DataFrame
            }
        }
        return NodePorts(inports=input_ports, outports=output_ports)

    def conf_schema(self):
        json = {
            "title": "PointNode configure",
            "type": "object",
            "properties": {
                "npts":  {
                    "type": "number",
                    "description": "number of data points",
                    "minimum": 10
                }
            },
            "required": ["npts"],
        }

        ui = {
            "npts": {"ui:widget": "updown"}
        }
        return ConfSchema(json=json, ui=ui)

    def init(self):
        pass

    def meta_setup(self):
        columns_out = {
            'points_df_out': {
                'x': 'float64',
                'y': 'float64'
            },
        }
        return MetaData(inports={}, outports=columns_out)

    def process(self, inputs):
        npts = self.conf['npts']
        df = pd.DataFrame()
        df['x'] = np.random.rand(npts)
        df['y'] = np.random.rand(npts)
        output = {}
        if self.outport_connected('points_df_out'):
            output.update({'points_df_out': df})
        return output
