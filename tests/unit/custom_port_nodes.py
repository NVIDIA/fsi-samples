import numpy as np
import cudf

from gquant.dataframe_flow import Node
from gquant.dataframe_flow import NodePorts, PortsSpecSchema


class PointNoPortsNode(Node):

    def columns_setup(self):
        self.required = {}
        self.addition = {
            'x': 'float64',
            'y': 'float64'
        }

    def process(self, inputs):
        npts = self.conf['npts']
        df = cudf.DataFrame()
        df['x'] = np.random.rand(npts)
        df['y'] = np.random.rand(npts)

        return df


class PointNode(Node):

    def ports_setup(self):
        input_ports = {}
        output_ports = {
            'points_df_out': {
                PortsSpecSchema.port_type: cudf.DataFrame
            }
        }

        return NodePorts(inports=input_ports, outports=output_ports)

    def columns_setup(self):
        self.required = {}
        self.addition = {
           'points_df_out': {
                'x': 'float64',
                'y': 'float64'
            }
        }

    def process(self, inputs):
        npts = self.conf['npts']
        seed = self.conf.get('nseed')
        if seed is not None:
            np.random.seed(seed)
        df = cudf.DataFrame()
        df['x'] = np.random.rand(npts)
        df['y'] = np.random.rand(npts)

        return {'points_df_out': df}


class DistanceNode(Node):

    def ports_setup(self):
        input_ports = {
            'points_df_in': {
                PortsSpecSchema.port_type: cudf.DataFrame
            }
        }

        output_ports = {
            'distance_df': {
                PortsSpecSchema.port_type: cudf.DataFrame
            }
        }

        return NodePorts(inports=input_ports, outports=output_ports)

    def columns_setup(self):
        self.delayed_process = True

        req_cols = {
            'x': 'float64',
            'y': 'float64'
        }

        self.required = {
            'points_df_in': req_cols,
            'distance_df': req_cols
        }

        self.addition = {
            'distance_df': {
                'distance_cudf': 'float64'
            }
        }

    def process(self, inputs):
        df = inputs['points_df_in']

        # DEBUGGING
        # try:
        #     from dask.distributed import get_worker
        #     worker = get_worker()
        #     print('worker{} process NODE "{}" worker: {}'.format(
        #         worker.name, self.uid, worker))
        # except (ValueError, ImportError):
        #     pass

        df['distance_cudf'] = (df['x'] ** 2 + df['y'] ** 2).sqrt()

        return {'distance_df': df}
