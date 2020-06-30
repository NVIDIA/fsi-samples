from gquant.dataframe_flow import Node
import datetime
import cudf
import dask_cudf
import xgboost as xgb
import dask


__all__ = ['XGBoostStrategyNode']


class XGBoostStrategyNode(Node):
    """
    This is the Node used to compute trading signal from XGBoost Strategy.
    It requires the following conf fields:
        "train_date": a date string of "Y-m-d" format. All the data points
        before this date is considered as training, otherwise as testing. If
        not provided, all the data points are considered as training.
        "xgboost_parameters": a dictionary of any legal parameters for XGBoost
        models. It overwrites the default parameters used in the process method
        "no_feature": specifying a list of columns in the input dataframe that
        should NOT be considered as training features.
        "target": the column that is considered as "target" in machine learning
        algorithm
    It requires the "datetime" column for spliting the data points and adds a
    new column "signal" to be used for backtesting.
    The detailed computation steps are listed in the process method's docstring
    """

    def columns_setup(self):
        self.required = {'datetime': 'date',
                         "asset": "int64"}
        self.retention = self.conf['no_feature']
        self.retention['signal'] = 'float64'

    def process(self, inputs):
        """
        The process is doing following things:
            1. split the data into training and testing based on provided
               conf['train_date']. If it is not provided, all the data is
               treated as training data.
            2. train a XGBoost model based on the training data
            3. Make predictions for all the data points including training and
               testing.
            4. From the prediction of returns, compute the trading signals that
               can be used in the backtesting.
        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        dxgb_params = {
                'max_depth':         8,
                'max_leaves':        2 ** 8,
                'tree_method':       'gpu_hist',
                'objective':         'reg:squarederror',
                'grow_policy':       'lossguide',
        }
        num_of_rounds = 100
        if 'xgboost_parameters' in self.conf:
            dxgb_params.update(self.conf['xgboost_parameters'])
        input_df = inputs[0]
        model_df = input_df
        train_cols = set(model_df.columns) - set(
            self.conf['no_feature'].keys())
        train_cols = list(train_cols - set([self.conf['target']]))

        if isinstance(input_df, dask_cudf.DataFrame):
            # get the client
            client = dask.distributed.client.default_client()
            if 'train_date' in self.conf:
                train_date = datetime.datetime.strptime(self.conf['train_date'],  # noqa: F841, E501
                                                        '%Y-%m-%d')
                model_df = model_df[model_df.datetime < train_date]
            train = model_df[train_cols]
            target = model_df[self.conf['target']]
            dmatrix = xgb.dask.DaskDMatrix(client, train, label=target)
            bst = xgb.dask.train(client, dxgb_params, dmatrix,
                                 num_boost_round=num_of_rounds)

            tree_booster = bst['booster']

            def predict(dask_df):
                cudf_df = dask_df
                infer_dmatrix = xgb.DMatrix(cudf_df[train_cols])
                prediction = cudf.Series(tree_booster.predict(infer_dmatrix),
                                         nan_as_null=False,
                                         index=cudf_df.index
                                         ).astype('float64')
                cudf_df['signal'] = prediction
                # here we need to remove the first day of prediction
                cudf_df['tmp'] = (cudf_df['asset'] -
                                  cudf_df['asset'].shift(1)).fillna(1)
                cudf_df['tmp'] = (cudf_df['tmp'] != 0).astype('int32')
                # cudf_df['tmp'][cudf_df['tmp'] == 1] = None
                tmp = cudf_df['tmp']
                cudf_df['tmp'] = tmp.where(tmp != 1, None)
                cudf_df = cudf_df.dropna(subset=['tmp'])
                cudf_df = cudf_df.drop('tmp')
                return cudf_df
            delayed_fun = dask.delayed(predict)
            delayedObj = [delayed_fun(dask_cudf.from_delayed(delayed)) for delayed in input_df.to_delayed()] # noqa E501
            input_df = dask_cudf.from_delayed(delayedObj)

        elif isinstance(input_df, cudf.DataFrame):
            if 'train_date' in self.conf:
                train_date = datetime.datetime.strptime(self.conf['train_date'],  # noqa: F841, E501
                                                        '%Y-%m-%d')
                model_df = model_df.query('datetime<@train_date')
            train = model_df[train_cols]
            target = model_df[self.conf['target']]
            dmatrix = xgb.DMatrix(train, label=target)
            bst = xgb.train(dxgb_params, dmatrix,
                            num_boost_round=num_of_rounds)
            # make inferences
            infer_dmatrix = xgb.DMatrix(input_df[train_cols])

            prediction = cudf.Series(bst.predict(infer_dmatrix),
                                     nan_as_null=False,
                                     index=input_df.index).astype('float64')
            input_df['signal'] = prediction
            # here we need to remove the first day of prediction
            input_df['tmp'] = (input_df['asset'] -
                               input_df['asset'].shift(1)).fillna(1)
            input_df['tmp'] = (input_df['tmp'] != 0).astype('int32')
            # input_df['tmp'][input_df['tmp'] == 1] = None
            tmp = input_df['tmp']
            input_df['tmp'] = tmp.where(tmp != 1, None)
            input_df = input_df.dropna(subset=['tmp'])
            input_df = input_df.drop('tmp')

        # convert the signal to trading action
        # 1 is buy and -1 is sell
        # It predicts the tomorrow's return (shift -1)
        # We shift 1 for trading actions so that it acts on the second day
        input_df['signal'] = ((
            input_df['signal'] >= 0).astype('float') * 2 - 1).shift(1)

        # remove the bad datapints
        input_df = input_df.dropna()
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
                       'datetime': 'date',
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
