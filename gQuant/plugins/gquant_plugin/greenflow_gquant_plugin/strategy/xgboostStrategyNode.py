from greenflow.dataframe_flow import Node
import datetime
import cudf
import dask_cudf
import xgboost as xgb
import dask
from .._port_type_node import _PortTypesMixin
from greenflow.dataframe_flow.portsSpecSchema import ConfSchema, MetaData


__all__ = ['XGBoostStrategyNode']


class XGBoostStrategyNode(_PortTypesMixin, Node):
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

    def init(self):
        _PortTypesMixin.init(self)
        self.INPUT_PORT_NAME = 'stock_in'
        self.OUTPUT_PORT_NAME = 'stock_out'

    def meta_setup(self):
        # if 'no_feature' in self.conf:
        #     retention = self.conf['no_feature']
        # else:
        cols_required = {'datetime': 'date',
                         "asset": "int64"}
        # self.delayed_process = True
        required = {
            self.INPUT_PORT_NAME: cols_required
        }
        retention = {}
        retention['signal'] = 'float64'
        # _PortTypesMixin.retention_meta_setup(self, retention)

        input_meta = self.get_input_meta()
        if self.INPUT_PORT_NAME not in input_meta:
            col_from_inport = required[self.INPUT_PORT_NAME]
        else:
            col_from_inport = input_meta[self.INPUT_PORT_NAME]
        # delete the columns from the inputs
        if 'no_feature' in self.conf:
            for key in self.conf['no_feature']:
                if key in col_from_inport:
                    retention[key] = col_from_inport[key]
        metadata = MetaData(inports=required,
                            outports={self.OUTPUT_PORT_NAME: retention})
        return metadata

    def ports_setup(self):
        types = [cudf.DataFrame,
                 dask_cudf.DataFrame]
        return _PortTypesMixin.ports_setup_from_types(self, types)

    def conf_schema(self):
        json = {
            "title": "XGBoost Node configure",
            "type": "object",
            "description": """Split the data into training and testing based on
             'train_data', train a XGBoost model based on the training data,
             make predictions for all the data points, compute the trading.
            """,
            "properties": {
                "num_of_rounds": {
                    "type": "number",
                    "description": """The number of rounds for boosting""",
                    "default": 100
                },
                "train_date":  {
                    "type": "string",
                    "description": """the date to splite train and validation
                    dataset"""
                },
                "target":  {
                    "type": "string",
                    "description": "the column used as dependent variable"
                },
                "no_feature": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                    "description": """columns in the input dataframe that
        should NOT be considered as training features."""
                },
                "xgboost_parameters": {
                    "type": "object",
                    "description": "xgoobst parameters",
                    "properties": {
                        'max_depth': {
                            "type": "number",
                            "description": "Maximum depth of a tree.",
                            "default": 8
                        },
                        "max_leaves": {
                            "type": "number",
                            "description": "maximum number of tree leaves",
                            "default": 2**8
                        },
                        "gamma": {
                            "type": "number",
                            "description": """Minimum loss reduction required
                            to make a further partition on a leaf node of the
                            tree.""",
                            "default": 0
                        },
                        "objective": {
                            "type": "string",
                            "enum": ["reg:squarederror", "reg:squaredlogerror",
                                     "reg:logistic", "reg:pseudohubererror"],
                            "description": """Specify the learning task and
                            the corresponding learning objective.""",
                            "default": "reg:squarederror"
                        }
                    }
                }
            },
            "required": ["target", "num_of_rounds"],
        }
        ui = {
            "train_date":  {
                "ui:widget": "alt-date",
                "ui:options": {
                        "yearsRange": [1985, 2025],
                        "hideNowButton": True,
                        "hideClearButton": True,
                },
            },
        }
        input_meta = self.get_input_meta()
        if self.INPUT_PORT_NAME in input_meta:
            col_from_inport = input_meta[self.INPUT_PORT_NAME]
            enums = [col for col in col_from_inport.keys()]
            json['properties']['no_feature']['items']['enum'] = enums
            json['properties']['target']['enum'] = enums
            return ConfSchema(json=json, ui=ui)
        else:
            return ConfSchema(json=json, ui=ui)

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
        # num_of_rounds = 100
        if 'xgboost_parameters' in self.conf:
            dxgb_params.update(self.conf['xgboost_parameters'])
        input_df = inputs[self.INPUT_PORT_NAME]
        model_df = input_df
        train_cols = set(model_df.columns) - set(
            self.conf['no_feature'])
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
                                 num_boost_round=self.conf["num_of_rounds"])

            dtrain = xgb.dask.DaskDMatrix(client, input_df[train_cols])
            prediction = xgb.dask.predict(client, bst, dtrain)
            pred_df = dask_cudf.from_dask_dataframe(
                prediction.to_dask_dataframe())
            pred_df.index = input_df.index
            input_df['signal'] = pred_df
        elif isinstance(input_df, cudf.DataFrame):
            if 'train_date' in self.conf:
                train_date = datetime.datetime.strptime(self.conf['train_date'],  # noqa: F841, E501
                                                        '%Y-%m-%d')
                model_df = model_df.query('datetime<@train_date')
            train = model_df[train_cols]
            target = model_df[self.conf['target']]
            dmatrix = xgb.DMatrix(train, label=target)
            bst = xgb.train(dxgb_params, dmatrix,
                            num_boost_round=self.conf["num_of_rounds"])
            infer_dmatrix = xgb.DMatrix(input_df[train_cols])
            prediction = cudf.Series(bst.predict(infer_dmatrix),
                                     nan_as_null=False,
                                     index=input_df.index
                                     ).astype('float64')
            input_df['signal'] = prediction

        input_df['tmp'] = (input_df['asset'] -
                           input_df['asset'].shift(1)).fillna(1)
        input_df['tmp'] = (input_df['tmp'] != 0).astype('int32')
        tmp = input_df['tmp']
        input_df['tmp'] = tmp.where(tmp != 1, None)
        input_df = input_df.dropna(subset=['tmp'])
        input_df = input_df.drop('tmp', axis=1)

        # convert the signal to trading action
        # 1 is buy and -1 is sell
        # It predicts the tomorrow's return (shift -1)
        # We shift 1 for trading actions so that it acts on the second day
        input_df['signal'] = ((
            input_df['signal'] >= 0).astype('float') * 2 - 1).shift(1)

        # remove the bad datapints
        input_df = input_df.dropna()
        remaining = list(self.conf['no_feature']) + ['signal']
        return {self.OUTPUT_PORT_NAME: input_df[remaining]}
