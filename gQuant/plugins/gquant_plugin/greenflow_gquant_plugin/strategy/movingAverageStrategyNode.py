from .. import cuindicator as ci
from greenflow.dataframe_flow import Node, PortsSpecSchema
from numba import cuda
import math
import numpy as np
import cudf
from greenflow.dataframe_flow.portsSpecSchema import ConfSchema
from greenflow.dataframe_flow.metaSpec import MetaDataSchema
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin
from ..node_hdf_cache import NodeHDFCacheMixin


@cuda.jit
def moving_average_signal_kernel(ma_fast, ma_slow, out_arr, arr_len):
    i = cuda.grid(1)
    if i == 0:
        out_arr[i] = np.nan
    if i < arr_len - 1:
        if math.isnan(ma_slow[i]) or math.isnan(ma_fast[i]):
            out_arr[i + 1] = np.nan
        elif ma_fast[i] - ma_slow[i] > 0.00001:
            # shift 1 time to make sure no peeking into the future
            out_arr[i + 1] = -1.0
        else:
            out_arr[i + 1] = 1.0


def moving_average_signal(stock_df, n_fast, n_slow):
    ma_slow = ci.moving_average(stock_df['close'],
                                n_slow).to_gpu_array()
    ma_fast = ci.moving_average(stock_df['close'],
                                n_fast).to_gpu_array()
    out_arr = cuda.device_array_like(ma_fast)
    array_len = len(ma_slow)
    number_of_threads = 256
    number_of_blocks = (array_len + (
        number_of_threads - 1)) // number_of_threads
    moving_average_signal_kernel[(number_of_blocks,),
                                 (number_of_threads,)](ma_fast,
                                                       ma_slow,
                                                       out_arr,
                                                       array_len)
    return out_arr, ma_slow, ma_fast


class MovingAverageStrategyNode(TemplateNodeMixin, NodeHDFCacheMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.INPUT_PORT_NAME = 'stock_in'
        self.OUTPUT_PORT_NAME = 'stock_out'
        self.delayed_process = True
        port_type = PortsSpecSchema.port_type
        port_inports = {
            self.INPUT_PORT_NAME: {
                port_type: [
                    "cudf.DataFrame",
                    "dask_cudf.DataFrame", "dask.dataframe.DataFrame"
                ]
            },
        }
        port_outports = {
            self.OUTPUT_PORT_NAME: {
                port_type: "${port:stock_in}"
            }
        }
        cols_required = {"close": "float64"}
        addition = {"signal": "float64",
                    "ma_slow": "float64",
                    "ma_fast": "float64"}
        meta_inports = {
            self.INPUT_PORT_NAME: cols_required
        }
        meta_outports = {
            self.OUTPUT_PORT_NAME: {
                MetaDataSchema.META_OP: MetaDataSchema.META_OP_ADDITION,
                MetaDataSchema.META_REF_INPUT: self.INPUT_PORT_NAME,
                MetaDataSchema.META_DATA: addition
            }
        }
        self.template_ports_setup(
            in_ports=port_inports,
            out_ports=port_outports
        )
        self.template_meta_setup(
            in_ports=meta_inports,
            out_ports=meta_outports
        )

    def conf_schema(self):
        json = {
            "title": "Moving Average Strategy Node configure",
            "type": "object",
            "description": """Simple mean reversion trading strategy.
            It computes two moving average signals of the `close`
            prices and decides long/short of asset when these two
             signals cross over.select the asset based on asset id""",
            "properties": {
                "fast":  {
                    "type": "number",
                    "description": "fast moving average window"
                },
                "slow":  {
                    "type": "number",
                    "description": "slow moving average window"
                }
            },
            "required": ["fast", "slow"],
        }
        ui = {
        }
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        Simple mean reversion trading strategy. It computes two moving average
        signals of the `close` prices and decides long/short of asset when
        these two signals cross over.

        The trading signal is named as `signal` in the dataframe. positive
        value means long and negative value means short. The resulting moving
        average signals are added to the dataframe.


        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        input_df = inputs[self.INPUT_PORT_NAME]
        n_fast = self.conf['fast']
        n_slow = self.conf['slow']
        signal, slow, fast = moving_average_signal(input_df, n_fast, n_slow)
        signal = cudf.Series(signal, index=input_df.index)
        slow = cudf.Series(slow, index=input_df.index)
        fast = cudf.Series(fast, index=input_df.index)
        input_df['signal'] = signal
        input_df['ma_slow'] = slow
        input_df['ma_slow'] = input_df['ma_slow'].fillna(0.0)
        input_df['ma_fast'] = fast
        input_df['ma_fast'] = input_df['ma_fast'].fillna(0.0)
        input_df = input_df.dropna()
        return {self.OUTPUT_PORT_NAME: input_df}
