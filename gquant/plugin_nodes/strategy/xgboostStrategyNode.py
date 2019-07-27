from gquant.dataframe_flow import Node
import datetime
import cudf
import xgboost as xgb
from numba import cuda
import math


@cuda.jit
def signal_kernel(signal_arr, out_arr, arr_len):
    i = cuda.grid(1)
    if i == 0:
        out_arr[i] = math.inf
    if i < arr_len - 1:
        if math.isnan(signal_arr[i]):
            out_arr[i + 1] = math.inf
        elif signal_arr[i] < 0.0:
            # shift 1 time to make sure no peeking into the future
            out_arr[i + 1] = -1.0
        else:
            out_arr[i + 1] = 1.0


def compute_signal(signal):
    signal_arr = signal.data.to_gpu_array()
    out_arr = cuda.device_array_like(signal_arr)
    number_of_threads = 256
    array_len = len(signal)
    number_of_blocks = (array_len + (
        number_of_threads - 1)) // number_of_threads
    signal_kernel[(number_of_blocks,),
                  (number_of_threads,)](signal_arr,
                                        out_arr,
                                        array_len)
    return out_arr


class XGBoostStrategyNode(Node):

    def columns_setup(self):
        self.required = {'datetime': 'datetime64[ms]'}
        self.retention = self.conf['no_feature']
        self.retention['signal'] = 'float64'

    def process(self, inputs):
        """
        Add technical indicators to the dataframe.
        All technical indicators are defined in the self.conf
        "remove_na" in self.conf decides whether we want to remove the NAs
        from the technical indicators

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        dxgb_params = {
                'nround':            100,
                'max_depth':         8,
                'max_leaves':        2 ** 8,
                'alpha':             0.9,
                'eta':               0.1,
                'gamma':             0.1,
                'learning_rate':     0.1,
                'subsample':         1,
                'reg_lambda':        1,
                'scale_pos_weight':  2,
                'min_child_weight':  30,
                'tree_method':       'gpu_hist',
                'n_gpus':            1,
                'distributed_dask':  True,
                'loss':              'ls',
                # 'objective':         'gpu:reg:linear',
                'objective':         'reg:squarederror',
                'max_features':      'auto',
                'criterion':         'friedman_mse',
                'grow_policy':       'lossguide',
                'verbose':           True
        }
        if 'xgboost_parameters' in self.conf:
            dxgb_params.update(self.conf['xgboost_parameters'])
        input_df = inputs[0]
        model_df = input_df
        if 'train_date' in self.conf:
            train_date = datetime.datetime.strptime(self.conf['train_date'],  # noqa: F841, E501
                                                    '%Y-%m-%d')
            model_df = model_df.query('datetime<@train_date')
        train_cols = set(model_df.columns) - set(
            self.conf['no_feature'].keys())
        train_cols = list(train_cols - set([self.conf['target']]))
        pd_model = model_df.to_pandas()
        train = pd_model[train_cols]
        target = pd_model[self.conf['target']]
        dmatrix = xgb.DMatrix(train, target)
        bst = xgb.train(dxgb_params, dmatrix,
                        num_boost_round=dxgb_params['nround'])
        # make inferences
        infer_dmatrix = xgb.DMatrix(input_df.to_pandas()[train_cols])
        prediction = cudf.Series(bst.predict(infer_dmatrix)).astype('float64')
        signal = compute_signal(prediction)
        input_df['signal'] = signal
        # remove the bad datapints
        input_df = input_df.query('signal<10')
        remaining = list(self.conf['no_feature'].keys()) + ['signal']
        return input_df[remaining]


if __name__ == "__main__":
    from gquant.plugin_nodes.dataloader.csvStockLoader import CsvStockLoader

    loader = CsvStockLoader("node_technical_indicator", {}, True, False)
    df = loader.load_cache('.cache'+'/'+loader.uid+'.hdf5')
    conf = {
        'train_date': '2010-1-1',
        'target': 'SHIFT_-1',
        'no_feature': {'asset': 'int64',
                       'datetime': 'datetime64[ms]',
                       'volume': 'float64',
                       'close': 'float64',
                       'open': 'float64',
                       'high': 'float64',
                       'low': 'float64',
                       'returns': 'float64',
                       'indicator': 'int32'}
    }
    inN = XGBoostStrategyNode("abc", conf)
    o = inN.process([df])
