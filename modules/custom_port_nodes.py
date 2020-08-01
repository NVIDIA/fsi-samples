import math
import numpy as np
from numba import cuda
import cupy
import cudf
import dask_cudf
import dask
import rmm
from gquant.dataframe_flow import Node
from gquant.dataframe_flow import NodePorts, PortsSpecSchema
from gquant.dataframe_flow import ConfSchema
import copy


def share_type(ports, connections):
    """
    utility function, the input/output type is determinted by the connected
    input port
    @params, ports, input or output port
    @connections, input port connection information, returned by
     self.get_connected_inports()
    return
        the finalized port types
    """
    ports_out = copy.deepcopy(ports)
    for key in ports:
        if key in connections:
            # connected
            types = connections[key]
            ports_out[key].update({PortsSpecSchema.port_type: types})
    return ports_out


class PointNode(Node):

    def ports_setup(self):
        input_ports = {}
        output_ports = {
            'points_df_out': {
                PortsSpecSchema.port_type: cudf.DataFrame
            },
            'points_ddf_out': {
                PortsSpecSchema.port_type: dask_cudf.DataFrame
            },
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
                },
                "npartitions":  {
                    "type": "number",
                    "description": "num of partitions in the Dask dataframe",
                    "minimum": 1
                }

            },
            "required": ["npts", "npartitions"],
        }

        ui = {
            "npts": {"ui:widget": "updown"},
            "npartitions": {"ui:widget": "updown"}
        }
        return ConfSchema(json=json, ui=ui)

    def init(self):
        self.required = {}

    def columns_setup(self):
        columns_out = {
            'points_df_out': {
                'x': 'float64',
                'y': 'float64'
            },
            'points_ddf_out': {
                'x': 'float64',
                'y': 'float64'
            }
        }
        return columns_out

    def process(self, inputs):
        npts = self.conf['npts']
        df = cudf.DataFrame()
        df['x'] = np.random.rand(npts)
        df['y'] = np.random.rand(npts)
        output = {}
        if self.outport_connected('points_df_out'):
            output.update({'points_df_out': df})
        if self.outport_connected('points_ddf_out'):
            npartitions = self.conf['npartitions']
            ddf = dask_cudf.from_cudf(df, npartitions=npartitions)
            output.update({'points_ddf_out': ddf})
        return output


class DistanceNode(Node):

    def ports_setup(self):
        port_type = PortsSpecSchema.port_type
        input_ports = {
            'points_df_in': {
                port_type: [cudf.DataFrame, dask_cudf.DataFrame]
            }
        }

        output_ports = {
            'distance_df': {
                port_type: [cudf.DataFrame, dask_cudf.DataFrame]
            }
        }
        input_connections = self.get_connected_inports()
        if 'points_df_in' in input_connections:
            types = input_connections['points_df_in']
            # connected, use the types passed in from parent
            return NodePorts(inports={'points_df_in': {port_type: types}},
                             outports={'distance_df': {port_type: types}})
        else:
            return NodePorts(inports=input_ports, outports=output_ports)

    def conf_schema(self):
        return ConfSchema()

    def init(self):
        self.delayed_process = True
        req_cols = {
            'x': 'float64',
            'y': 'float64'
        }
        self.required = {
            'points_df_in': req_cols,
            'distance_df': req_cols
        }

    def columns_setup(self):
        input_columns = self.get_input_columns()
        col_from_inport = input_columns['points_df_in']
        # additional ports
        output_cols = {
            'distance_df': {
                'distance_cudf': 'float64'
            }
        }
        output_cols['distance_df'].update(col_from_inport)
        return output_cols

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
        port_type = PortsSpecSchema.port_type
        input_ports = {
            'points_df_in': {
                port_type: [cudf.DataFrame,
                            dask_cudf.DataFrame]
            }
        }

        output_ports = {
            'distance_df': {
                port_type: [cudf.DataFrame,
                            dask_cudf.DataFrame]
            }
        }

        input_connections = self.get_connected_inports()
        if 'points_df_in' in input_connections:
            types = input_connections['points_df_in']
            # connected
            return NodePorts(inports={'points_df_in': {port_type: types}},
                             outports={'distance_df': {port_type: types}})
        else:
            return NodePorts(inports=input_ports, outports=output_ports)
  
    def init(self):
        self.delayed_process = True
        required = {'x': 'float64',
                    'y': 'float64'}
        self.required = {
            'points_df_in': required,
            'distance_df': required
        }

    def columns_setup(self,):
        input_columns = self.get_input_columns()
        col_from_inport = input_columns['points_df_in']
        # additional ports
        output_cols = {
            'distance_df': {'distance_numba': 'float64'}
        }
        output_cols['distance_df'].update(col_from_inport)
        return output_cols

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
        darr = rmm.device_array(len(df))
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
        port_type = PortsSpecSchema.port_type
        input_ports = {
            'points_df_in': {
                port_type: [cudf.DataFrame,
                            dask_cudf.DataFrame]
            }
        }

        output_ports = {
            'distance_df': {
                port_type: [cudf.DataFrame,
                            dask_cudf.DataFrame]
            }
        }

        input_connections = self.get_connected_inports()
        if 'points_df_in' in input_connections:
            types = input_connections['points_df_in']
            # connected
            return NodePorts(inports={'points_df_in': {port_type: types}},
                             outports={'distance_df': {port_type: types}})
        else:
            return NodePorts(inports=input_ports, outports=output_ports)

    def init(self):
        self.delayed_process = True
        cols_required = {'x': 'float64',
                         'y': 'float64'}
        self.required = {
            'points_df_in': cols_required,
            'distance_df': cols_required
        }

    def columns_setup(self,):
        input_columns = self.get_input_columns()
        col_from_inport = input_columns['points_df_in']
        # additional ports
        output_cols = {
            'distance_df': {
                'distance_cupy': 'float64'
            }
        }
        output_cols['distance_df'].update(col_from_inport)
        return output_cols

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

    def init(self):
        required = {
            'x': 'float64',
            'y': 'float64'
        }

        self.required = {
            'points_df_in': required,
            'points_ddf_out': required
        }

    def columns_setup(self,):
        input_columns = self.get_input_columns()
        col_from_inport = input_columns['points_df_in']
        # additional ports
        output_cols = {
            'points_ddf_out': col_from_inport
        }
        return output_cols

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

        input_connections = self.get_connected_inports()
        input_ports_out = share_type(input_ports, input_connections)
        return NodePorts(inports=input_ports_out, outports=output_ports)

    def columns_setup(self):
        return {'max_diff': {}}

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
        max_difference = float(max_difference)
        # print('Max Difference: {}'.format(max_difference))
        # assert(max_difference < 1e-8)

        return {'max_diff': max_difference}
