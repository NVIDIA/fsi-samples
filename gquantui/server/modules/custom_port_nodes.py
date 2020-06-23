import math
import numpy as np
from numba import cuda
import cupy
import cudf
import dask_cudf
import dask

from gquant.dataframe_flow import Node
from gquant.dataframe_flow import NodePorts, PortsSpecSchema
from gquant.dataframe_flow import ConfSchema


class PointNode(Node):

    def ports_setup(self):
        input_ports = {}
        output_ports = {
            'points_df_out': {
                PortsSpecSchema.port_type: cudf.DataFrame
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
        df = cudf.DataFrame()
        df['x'] = np.random.rand(npts)
        df['y'] = np.random.rand(npts)

        return {'points_df_out': df}


class DistanceNode(Node):

    def ports_setup(self):
        input_ports = {
            'points_df_in': {
                PortsSpecSchema.port_type: [cudf.DataFrame,
                                            dask_cudf.DataFrame]
            }
        }

        output_ports = {
            'distance_df': {
                PortsSpecSchema.port_type: [cudf.DataFrame,
                                            dask_cudf.DataFrame]
            }
        }

        return NodePorts(inports=input_ports, outports=output_ports)

    def conf_schema(self):
        return ConfSchema()

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


@cuda.jit
def distance_kernel(x, y, distance, array_len):
    # ii - overall thread index
    ii = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if ii < array_len:
        distance[ii] = math.sqrt(x[ii] ** 2 + y[ii] ** 2)


class NumbaDistanceNode(Node):

    def ports_setup(self):
        input_ports = {
            'points_df_in': {
                PortsSpecSchema.port_type: [cudf.DataFrame,
                                            dask_cudf.DataFrame]
            }
        }

        output_ports = {
            'distance_df': {
                PortsSpecSchema.port_type: [cudf.DataFrame,
                                            dask_cudf.DataFrame]
            }
        }

        return NodePorts(inports=input_ports, outports=output_ports)

    def columns_setup(self,):
        self.delayed_process = True

        required = {'x': 'float64',
                    'y': 'float64'}
        self.required = {
            'points_df_in': required,
            'distance_df': required
        }
        self.addition = {
            'distance_df': {'distance_numba': 'float64'}
        }

    def conf_schema(self):
        return ConfSchema()

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

        number_of_threads = 16
        number_of_blocks = ((len(df) - 1) // number_of_threads) + 1
        # Inits device array by setting 0 for each index.
        # df['distance_numba'] = 0.0
        darr = cuda.device_array(len(df))
        distance_kernel[(number_of_blocks,), (number_of_threads,)](
            df['x'],
            df['y'],
            darr,
            len(df))
        df['distance_numba'] = darr
        return {'distance_df': df}


kernel_string = r'''
    extern "C" __global__
    void compute_distance(const double* x, const double* y,
            double* distance, int arr_len) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < arr_len){
        distance[tid] = sqrt(x[tid]*x[tid] + y[tid]*y[tid]);
        }
    }
'''


class CupyDistanceNode(Node):

    def ports_setup(self):
        input_ports = {
            'points_df_in': {
                PortsSpecSchema.port_type:[cudf.DataFrame,
                                            dask_cudf.DataFrame]
            }
        }

        output_ports = {
            'distance_df': {
                PortsSpecSchema.port_type:[cudf.DataFrame,
                                            dask_cudf.DataFrame]
            }
        }

        return NodePorts(inports=input_ports, outports=output_ports)

    def columns_setup(self,):
        cols_required = {'x': 'float64',
                         'y': 'float64'}
        self.required = {
            'points_df_in': cols_required,
            'distance_df': cols_required
        }

        self.addition = {
            'distance_df': {
                'distance_cupy': 'float64'
            }
        }
        self.delayed_process = True

    def conf_schema(self):
        return ConfSchema()

    def get_kernel(self):
        raw_kernel = cupy.RawKernel(kernel_string, 'compute_distance')
        return raw_kernel

    def process(self, inputs):
        df = inputs['points_df_in']
        cupy_x = cupy.asarray(df['x'])
        cupy_y = cupy.asarray(df['y'])
        number_of_threads = 16
        number_of_blocks = (len(df) - 1) // number_of_threads + 1
        dis = cupy.ndarray(len(df), dtype=cupy.float64)
        self.get_kernel()((number_of_blocks,), (number_of_threads,),
                          (cupy_x, cupy_y, dis, len(df)))
        df['distance_cupy'] = dis

        return {'distance_df': df}


class DistributedNode(Node):

    def ports_setup(self):
        input_ports = {
            'points_df_in': {
                PortsSpecSchema.port_type: cudf.DataFrame
            }
        }

        output_ports = {
            'points_ddf_out': {
                PortsSpecSchema.port_type: dask_cudf.DataFrame
            }
        }

        return NodePorts(inports=input_ports, outports=output_ports)

    def columns_setup(self,):
        required = {
            'x': 'float64',
            'y': 'float64'
        }

        self.required = {
            'points_df_in': required,
            'points_ddf_out': required
        }

    def conf_schema(self):
        json = {
            "title": "DistributedNode configure",
            "type": "object",
            "properties": {
                "npartitions":  {
                    "type": "number",
                    "description": "num of partitions in the Dask dataframe",
                    "minimum": 1
                }
            },
            "required": ["npartitions"],
        }

        ui = {
            "npartitions": {"ui:widget": "updown"}
        }
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        npartitions = self.conf['npartitions']
        df = inputs['points_df_in']
        ddf = dask_cudf.from_cudf(df, npartitions=npartitions)
        return {'points_ddf_out': ddf}


class VerifyNode(Node):

    def ports_setup(self):
        input_ports = {
            'df1': {
                PortsSpecSchema.port_type: [cudf.DataFrame,
                                            dask_cudf.DataFrame]
            },
            'df2': {
                PortsSpecSchema.port_type: [cudf.DataFrame,
                                            dask_cudf.DataFrame]
            }
        }
        output_ports = {
            'max_diff': {
                PortsSpecSchema.port_type: float
            }
        }

        return NodePorts(inports=input_ports, outports=output_ports)

    def columns_setup(self):
        pass

    def conf_schema(self):
        json = {
            "title": "VerifyNode configure",
            "type": "object",
            "properties": {
                "df1_col":  {
                    "type": "string",
                    "description": "dataframe1 column name"
                },
                "df2_col":  {
                    "type": "string",
                    "description": "dataframe2 column name"
                }
            },
            "required": ["df1_col", "df2_col"],
        }

        ui = {
            "df1_col": {"ui:widget": "text"},
            "df2_col": {"ui:widget": "text"}
        }
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        df1 = inputs['df1']
        df2 = inputs['df2']
        col_df1 = self.conf['df1_col']
        col_df2 = self.conf['df2_col']

        df1_col = df1[col_df1]
        if isinstance(df1, dask_cudf.DataFrame):
            # df1_col = df1_col.compute()
            pass

        df2_col = df2[col_df2]
        if isinstance(df2, dask_cudf.DataFrame):
            # df2_col = df2_col.compute()
            pass

        max_difference = (df1_col - df2_col).abs().max()

        if isinstance(max_difference, dask.dataframe.core.Scalar):
            max_difference = float(max_difference.compute())

        # print('Max Difference: {}'.format(max_difference))
        # assert(max_difference < 1e-8)

        return {'max_diff': max_difference}
