import gquant.cuindicator as ci
from gquant.dataframe_flow import Node
from numba import cuda
import math


@cuda.jit
def moving_average_signal_kernel(ma_fast, ma_slow, out_arr, arr_len):
    i = cuda.grid(1)
    if i == 0:
        out_arr[i] = math.inf
    if i < arr_len - 1:
        if math.isnan(ma_slow[i]) or math.isnan(ma_fast[i]):
            out_arr[i + 1] = math.inf
        elif ma_fast[i] - ma_slow[i] > 0.00001:
            # shift 1 time to make sure no peeking into the future
            out_arr[i + 1] = -1.0
        else:
            out_arr[i + 1] = 1.0


def moving_average_signal(stock_df, n_fast, n_slow):
    ma_slow = ci.moving_average(stock_df['close'],
                                n_slow).data.to_gpu_array()
    ma_fast = ci.moving_average(stock_df['close'],
                                n_fast).data.to_gpu_array()
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


class MovingAverageStrategyNode(Node):

    def columns_setup(self):
        self.required = {"close": "float64"}
        self.addition = {"signal": "float64",
                         "ma_slow": "float64",
                         "ma_fast": "float64"}

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
        input_df = inputs[0]
        n_fast = self.conf['fast']
        n_slow = self.conf['slow']
        signal, slow, fast = moving_average_signal(input_df, n_fast, n_slow)
        input_df['signal'] = signal
        input_df['ma_slow'] = slow
        input_df['ma_slow'] = input_df['ma_slow'].fillna(0.0)
        input_df['ma_fast'] = fast
        input_df['ma_fast'] = input_df['ma_fast'].fillna(0.0)
        input_df = input_df.query('signal<10')  # remove the bad datapints
        return input_df

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
