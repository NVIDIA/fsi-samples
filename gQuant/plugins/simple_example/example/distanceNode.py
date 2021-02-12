import pandas as pd
import numpy as np
from gquant.dataframe_flow import Node, MetaData
from gquant.dataframe_flow import NodePorts, PortsSpecSchema
from gquant.dataframe_flow import ConfSchema


class DistanceNode(Node):

    def ports_setup(self):
        port_type = PortsSpecSchema.port_type
        input_ports = {
            'points_df_in': {
                port_type: [pd.DataFrame]
            }
        }

        output_ports = {
            'distance_df': {
                port_type: [pd.DataFrame]
            },
            'distance_abs_df': {
                PortsSpecSchema.port_type:  [pd.DataFrame]
            }
        }
        input_connections = self.get_connected_inports()
        if 'points_df_in' in input_connections:
            types = input_connections['points_df_in']
            # connected, use the types passed in from parent
            return NodePorts(inports={'points_df_in': {port_type: types}},
                             outports={'distance_df': {port_type: types},
                                       'distance_abs_df': {port_type: types},
                                       })
        else:
            return NodePorts(inports=input_ports, outports=output_ports)

    def conf_schema(self):
        return ConfSchema()

    def init(self):
        self.delayed_process = True

    def meta_setup(self):
        req_cols = {
            'x': 'float64',
            'y': 'float64'
        }
        required = {
            'points_df_in': req_cols,
        }
        input_meta = self.get_input_meta()
        output_cols = ({
                'distance_df': {
                    'distance_cudf': 'float64',
                    'x': 'float64',
                    'y': 'float64'
                },
                'distance_abs_df': {
                    'distance_abs_cudf': 'float64',
                    'x': 'float64',
                    'y': 'float64'
                }
            })
        if 'points_df_in' in input_meta:
            col_from_inport = input_meta['points_df_in']
            # additional ports
            output_cols['distance_df'].update(col_from_inport)
            output_cols['distance_abs_df'].update(col_from_inport)
        return MetaData(inports=required, outports=output_cols)

    def process(self, inputs):
        df = inputs['points_df_in']
        output = {}
        if self.outport_connected('distance_df'):
            copy_df = df.copy()
            copy_df['distance_cudf'] = np.sqrt((df['x'] ** 2 + df['y'] ** 2))
            output.update({'distance_df': copy_df})
        if self.outport_connected('distance_abs_df'):
            copy_df = df.copy()
            copy_df['distance_abs_cudf'] = np.abs(df['x']) + np.abs(df['y'])
            output.update({'distance_abs_df': copy_df})
        return output
