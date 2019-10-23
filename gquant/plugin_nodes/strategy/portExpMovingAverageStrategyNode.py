from gquant.dataframe_flow import Node
from .movingAverageStrategyNode import MovingAverageStrategyNode
import gquant.cuindicator as ci
from numba import cuda
from functools import partial
import math
import numpy as np


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


def port_exponential_moving_average(stock_df, n_fast, n_slow):
    ma_slow = ci.port_exponential_moving_average(stock_df['indicator'],
                                                 stock_df['close'],
                                                 n_slow).data.to_gpu_array()
    ma_fast = ci.port_exponential_moving_average(stock_df['indicator'],
                                                 stock_df['close'],
                                                 n_fast).data.to_gpu_array()
    out_arr = cuda.device_array_like(ma_fast)
    number_of_threads = 256
    array_len = len(stock_df)
    number_of_blocks = (array_len + (
        number_of_threads - 1)) // number_of_threads
    moving_average_signal_kernel[(number_of_blocks,),
                                 (number_of_threads,)](ma_fast,
                                                       ma_slow,
                                                       out_arr,
                                                       array_len)
    return out_arr, ma_slow, ma_fast


class PortExpMovingAverageStrategyNode(Node):

    def columns_setup(self):
        self.delayed_process = True
        self.required = {"close": "float64",
                         "indicator": "int32"}
        self.addition = {"signal": "float64",
                         "exp_ma_slow": "float64",
                         "exp_ma_fast": "float64"}

    def process(self, inputs):
        """
        Simple mean reversion trading strategy. It computes two exponential
        moving average signals of the `close` prices and decides long/short
        of asset when these two signals cross over. It computes the trading
        signals for all the assets in the dataframe.

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

        input_df = inputs[0]
        n_fast = self.conf['fast']
        n_slow = self.conf['slow']
        signal, slow, fast = port_exponential_moving_average(input_df,
                                                             n_fast,
                                                             n_slow)
        input_df['signal'] = signal
        input_df['exp_ma_slow'] = slow
        input_df['exp_ma_slow'] = input_df['exp_ma_slow'].fillna(0.0)
        input_df['exp_ma_fast'] = fast
        input_df['exp_ma_fast'] = input_df['exp_ma_fast'].fillna(0.0)
        # remove the bad datapints
        input_df = input_df.dropna()
        input_df = input_df.query('indicator == 0')
        return input_df


def cpu_exp_moving_average(df, n_slow=1, n_fast=3):
    df['exp_ma_slow'] = df['close'].ewm(span=n_slow, min_periods=n_slow).mean()
    df['exp_ma_fast'] = df['close'].ewm(span=n_fast, min_periods=n_fast).mean()
    df['signal'] = 1.0
    df['signal'][df['exp_ma_fast'] - df['exp_ma_slow'] > 0.00001] = -1.0
    df['signal'] = df['signal'].shift(1)
    df['signal'].values[0:n_slow] = np.nan
    return df


class CpuPortExpMovingAverageStrategyNode(PortExpMovingAverageStrategyNode):

    def process(self, inputs):
        """
        Simple mean reversion trading strategy. It computes two exponential
        moving average signals of the `close` prices and decides long/short
        of asset when these two signals cross over. It computes the trading
        signals for all the assets in the dataframe.

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
        input_df = inputs[0]
        n_fast = self.conf['fast']
        n_slow = self.conf['slow']
        fun = partial(cpu_exp_moving_average, n_fast=n_fast, n_slow=n_slow)
        input_df = input_df.groupby("asset").apply(fun)
        return input_df.dropna(subset=['signal'])

if __name__ == "__main__":
    from gquant.dataloader.csvStockLoader import CsvStockLoader
    from gquant.transform.assetFilterNode import AssetFilterNode
    from gquant.transform.sortNode import SortNode

    loader = CsvStockLoader("node_csvdata", {}, True, False)
    df = loader([])
    sf = AssetFilterNode("id2", {"asset": 22123})
    df2 = sf([df])
    sf2 = SortNode("id3", {"keys": ["asset", 'datetime']})
    df3 = sf2([df2])
    sf3 = MovingAverageStrategyNode('id4', {'fast': 5, 'slow': 10})
    df4 = sf3([df3])
