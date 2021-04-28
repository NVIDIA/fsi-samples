from greenflow.dataframe_flow import Node
import cudf
import dask_cudf
import xgboost as xgb
import dask
from greenflow.dataframe_flow.portsSpecSchema import (ConfSchema,
                                                      PortsSpecSchema)
from greenflow.dataframe_flow.metaSpec import MetaDataSchema
import copy
from collections import OrderedDict
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin
from ..node_hdf_cache import NodeHDFCacheMixin


__all__ = ['TrainXGBoostNode', 'InferXGBoostNode']


class TrainXGBoostNode(TemplateNodeMixin, NodeHDFCacheMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.INPUT_PORT_NAME = 'in'
        self.OUTPUT_PORT_NAME = 'model_out'
        port_type = PortsSpecSchema.port_type
        port_inports = {
            self.INPUT_PORT_NAME: {
                port_type: [
                    "cudf.DataFrame",
                    "dask_cudf.DataFrame", "dask.dataframe.DataFrame"
                ]
            }
        }
        port_outports = {
            self.OUTPUT_PORT_NAME: {
                port_type: ['xgboost.Booster', 'builtins.dict']
            }
        }

        cols_required = {}
        if 'columns' in self.conf and self.conf.get('include', True):
            cols_required = {}
            for col in self.conf['columns']:
                cols_required[col] = None
        meta_inports = {
            self.INPUT_PORT_NAME: cols_required,
        }
        meta_outports = {
            self.OUTPUT_PORT_NAME: {
                MetaDataSchema.META_OP: MetaDataSchema.META_OP_RETENTION,
                MetaDataSchema.META_DATA: {}
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

    def update(self):
        TemplateNodeMixin.update(self)
        input_meta = self.get_input_meta()
        if self.INPUT_PORT_NAME in input_meta:
            meta_inports = self.template_meta_setup().inports
            meta_outports = self.template_meta_setup().outports
            col_from_inport = input_meta[self.INPUT_PORT_NAME]
            enums = [col for col in col_from_inport.keys()]
            cols_output = {}
            cols_output['train'] = OrderedDict()
            cols_output['label'] = OrderedDict()
            if 'columns' in self.conf:
                if self.conf.get('include', True):
                    included_colums = self.conf['columns']
                else:
                    included_colums = [col for col in enums
                                       if col not in self.conf['columns']]
                cols_required = {}
                for col in included_colums:
                    if col in col_from_inport:
                        cols_required[col] = col_from_inport[col]
                        cols_output['train'][col] = col_from_inport[col]
                    else:
                        cols_required[col] = None
                        cols_output['train'][col] = None
                if ('target' in self.conf and
                        self.conf['target'] in col_from_inport):
                    cols_required[self.conf['target']
                                  ] = col_from_inport[self.conf['target']]
                    cols_output['label'][
                        self.conf['target']] = col_from_inport[
                            self.conf['target']]
                else:
                    cols_required[self.conf['target']] = None
                    cols_output['label'][
                        self.conf['target']] = None
                meta_inports[self.INPUT_PORT_NAME] = cols_required
            meta_outports[self.OUTPUT_PORT_NAME][
                MetaDataSchema.META_DATA] = cols_output
            self.template_meta_setup(
                in_ports=meta_inports,
                out_ports=meta_outports
            )

    def conf_schema(self):
        json = {
            "title": "XGBoost Node configure",
            "type": "object",
            "description": """train a XGBoost model for the input data,
            """,
            "properties": {
                "num_of_rounds": {
                    "type": "number",
                    "description": """The number of rounds for boosting""",
                    "default": 100
                },
                "target":  {
                    "type": "string",
                    "description": "the column used as dependent variable"
                },
                "columns": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                    "description": """columns in the input dataframe that are
                    considered as training features or not depending on
                    `include` flag."""
                },
                "include":  {
                    "type": "boolean",
                    "description": """if set true, the `columns` are treated
                    as independent variables.  if false, all dataframe columns
                    are independent variables except the `columns`""",
                    "default": True
                },
                "xgboost_parameters": {
                    "type": "object",
                    "description": "xgoobst parameters",
                    "properties": {
                        'eta': {
                            "type": "number",
                            "description": """Step size shrinkage used in
                            update to prevents overfitting. After each boosting
                            step, we can directly get the weights of new
                            features, and eta shrinks the feature weights to
                            make the boosting process more conservative.""",
                            "default": 0.3
                        },
                        'min_child_weight': {
                            "type": "number",
                            "description": """Minimum sum of instance weight
                            (hessian) needed in a child. If the tree partition
                            step results in a leaf node with the sum of
                            instance weight less than min_child_weight,
                            then the building process will give up further
                            partitioning. In linear regression task, this
                            simply corresponds to minimum number of instances
                            needed to be in each node. The larger
                            min_child_weight is, the more conservative
                            the algorithm will be.""",
                            "default": 1.0
                        },
                        'subsample': {
                            "type": "number",
                            "description": """Subsample ratio of the training
                            instances. Setting it to 0.5 means that XGBoost
                            would randomly sample half of the training data
                            prior to growing trees. and this will prevent
                            overfitting. Subsampling will occur once in every
                            boosting iteration.""",
                            "default": 1.0
                        },
                        'sampling_method': {
                            "type": "string",
                            "description": """The method to use to sample the
                            training instances.""",
                            "enum": ["uniform", "gradient_based"],
                            "default": "uniform",
                        },
                        'colsample_bytree': {
                            "type": "number",
                            "description": """is the subsample ratio of
                            columns when constructing each tree. Subsampling
                            occurs once for every tree constructed.""",
                            "default": 1.0
                        },
                        'colsample_bylevel': {
                            "type": "number",
                            "description": """is the subsample ratio of columns
                            for each level. Subsampling occurs once for every
                            new depth level reached in a tree. Columns are
                            subsampled from the set of columns chosen for the
                            current tree""",
                            "default": 1.0
                        },
                        'colsample_bynode': {
                            "type": "number",
                            "description": """is the subsample ratio of
                            columns for each node (split). Subsampling occurs
                            once every time a new split is evaluated. Columns
                            are subsampled from the set of columns chosen for
                            the current level.""",
                            "default": 1.0
                        },
                        'max_depth': {
                            "type": "integer",
                            "description": "Maximum depth of a tree.",
                            "default": 8
                        },
                        "max_leaves": {
                            "type": "integer",
                            "description": "maximum number of tree leaves",
                            "default": 2**8
                        },
                        "grow_policy": {
                            "type": "string",
                            "enum": ["depthwise", "lossguide"],
                            "description": """Controls a way new nodes are
                            added to the tree. Currently supported only if
                            tree_method is set to hist.""",
                            "default": "depthwise"
                        },
                        "gamma": {
                            "type": "number",
                            "description": """Minimum loss reduction required
                            to make a further partition on a leaf node of the
                            tree.""",
                            "default": 0.0
                        },
                        "lambda": {
                            "type": "number",
                            "description": """L2 regularization term on
                            weights. Increasing this value will make model
                            more conservative.""",
                            "default": 1.0
                        },
                        "alpha": {
                            "type": "number",
                            "description": """L1 regularization term on
                            weights. Increasing this value will make model more
                            conservative.""",
                            "default": 0.0
                        },
                        "tree_method": {
                            "type": "string",
                            "description": """The tree construction algorithm
                            used in XGBoost""",
                            "enum": ["auto", "exact", "approx", 'hist',
                                     'gpu_hist'],
                            "default": "auto"
                        },
                        "single_precision_histogram": {
                            "type": "boolean",
                            "description": """for hist and `gpu_hist tree
                             method, Use single precision to build histograms
                             instead of double precision.""",
                            "default": False
                        },
                        "deterministic_histogram": {
                            "type": "boolean",
                            "description": """for gpu_hist tree method, Build
                            histogram on GPU deterministically. Histogram
                            building is not deterministic due to the
                            non-associative aspect of floating point summation.
                            We employ a pre-rounding routine to mitigate the
                            issue, which may lead to slightly lower accuracy.
                            Set to false to disable it.""",
                            "default": False
                        },
                        "objective": {
                            "type": "string",
                            "enum": ["reg:squarederror", "reg:squaredlogerror",
                                     "reg:logistic", "reg:pseudohubererror",
                                     "binary:logistic", "binary:logitraw",
                                     "binary:hinge", "count:poisson",
                                     "survival:cox", "survival:aft",
                                     "aft_loss_distribution", "multi:softmax",
                                     "multi:softprob", "rank:pairwise",
                                     "rank:ndcg", "rank:map", "reg:gamma",
                                     "reg:tweedie"
                                     ],
                            "description": """Specify the learning task and
                            the corresponding learning objective.""",
                            "default": "reg:squarederror"
                        }
                    }
                }
            },
            "required": [],
        }
        ui = {}
        input_meta = self.get_input_meta()
        if self.INPUT_PORT_NAME in input_meta:
            col_from_inport = input_meta[self.INPUT_PORT_NAME]
            enums = [col for col in col_from_inport.keys()]
            json['properties']['columns']['items']['enum'] = enums
            json['properties']['target']['enum'] = enums
            return ConfSchema(json=json, ui=ui)
        else:
            return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        dxgb_params = {
                'max_depth':         8,
                'max_leaves':        2 ** 8,
                'tree_method':       'gpu_hist',
                'objective':         'reg:squarederror',
        }
        # num_of_rounds = 100
        if 'xgboost_parameters' in self.conf:
            dxgb_params.update(self.conf['xgboost_parameters'])
        input_df = inputs[self.INPUT_PORT_NAME]
        if self.conf.get('include', True):
            included_colums = self.conf['columns']
        else:
            included_colums = [col for col in input_df.columns
                               if col not in self.conf['columns']]
        train_cols = [col for col in included_colums
                      if col != self.conf['target']]
        # train_cols.sort()

        if isinstance(input_df, dask_cudf.DataFrame):
            # get the client
            client = dask.distributed.client.default_client()
            train = input_df[train_cols]
            target = input_df[self.conf['target']]
            dmatrix = xgb.dask.DaskDMatrix(client, train, label=target)
            bst = xgb.dask.train(client, dxgb_params, dmatrix,
                                 num_boost_round=self.conf["num_of_rounds"])
        elif isinstance(input_df, cudf.DataFrame):
            train = input_df[train_cols]
            target = input_df[self.conf['target']]
            dmatrix = xgb.DMatrix(train, label=target)
            bst = xgb.train(dxgb_params, dmatrix,
                            num_boost_round=self.conf["num_of_rounds"])
        return {self.OUTPUT_PORT_NAME: bst}


class InferXGBoostNode(TemplateNodeMixin, NodeHDFCacheMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.INPUT_PORT_NAME = 'data_in'
        self.INPUT_PORT_MODEL_NAME = 'model_in'
        self.OUTPUT_PORT_NAME = 'out'
        port_type = PortsSpecSchema.port_type
        port_inports = {
            self.INPUT_PORT_NAME: {
                port_type: [
                    "cudf.DataFrame",
                    "dask_cudf.DataFrame", "dask.dataframe.DataFrame"
                ]
            },
            self.INPUT_PORT_MODEL_NAME: {
                port_type: ['xgboost.Booster', 'builtins.dict']
            },
        }
        port_outports = {
            self.OUTPUT_PORT_NAME: {
                port_type: "${port:data_in}"
            }
        }

        meta_inports = {
            self.INPUT_PORT_NAME: {},
            self.INPUT_PORT_MODEL_NAME: {}
        }
        predict = self.conf.get('prediction', 'predict')
        out_cols = {predict: None}
        meta_outports = {
            self.OUTPUT_PORT_NAME: {
                MetaDataSchema.META_OP: MetaDataSchema.META_OP_RETENTION,
                MetaDataSchema.META_DATA: out_cols
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

    def update(self):
        TemplateNodeMixin.update(self)
        input_meta = self.get_input_meta()
        predict = self.conf.get('prediction', 'predict')
        pred_contribs: bool = self.conf.get('pred_contribs', False)
        meta_inports = self.template_meta_setup().inports
        meta_outports = self.template_meta_setup().outports
        if (self.INPUT_PORT_NAME in input_meta
                and self.INPUT_PORT_MODEL_NAME in input_meta):
            col_from_inport = input_meta[self.INPUT_PORT_NAME]
            if 'train' in input_meta[self.INPUT_PORT_MODEL_NAME]:
                required_cols = input_meta[
                    self.INPUT_PORT_MODEL_NAME]['train']
            else:
                required_cols = {}
            if not pred_contribs:
                col_from_inport[predict] = None  # the type is not determined
            else:
                col_from_inport = {}
                for i in range(len(required_cols)+1):
                    col_from_inport[i] = None
            meta_inports[self.INPUT_PORT_NAME] = required_cols
            meta_outports[self.OUTPUT_PORT_NAME][
                MetaDataSchema.META_DATA] = col_from_inport
        elif (self.INPUT_PORT_NAME not in input_meta and
              self.INPUT_PORT_MODEL_NAME in input_meta):
            if 'train' in input_meta[self.INPUT_PORT_MODEL_NAME]:
                required_cols = input_meta[
                    self.INPUT_PORT_MODEL_NAME]['train']
            else:
                required_cols = {}
            col_from_inport = copy.copy(required_cols)
            if not pred_contribs:
                col_from_inport[predict] = None  # the type is not determined
            else:
                col_from_inport = {}
                for i in range(len(required_cols)+1):
                    col_from_inport[i] = None
            meta_inports[self.INPUT_PORT_NAME] = required_cols
            meta_outports[self.OUTPUT_PORT_NAME][
                MetaDataSchema.META_DATA] = col_from_inport
        elif (self.INPUT_PORT_NAME in input_meta and
              self.INPUT_PORT_MODEL_NAME not in input_meta):
            col_from_inport = input_meta[self.INPUT_PORT_NAME]
            if not pred_contribs:
                col_from_inport[predict] = None  # the type is not determined
            meta_outports[self.OUTPUT_PORT_NAME][
                MetaDataSchema.META_DATA] = col_from_inport
        self.template_meta_setup(
            in_ports=meta_inports,
            out_ports=meta_outports
        )

    def conf_schema(self):
        json = {
            "title": "XGBoost Inference Node configure",
            "type": "object",
            "description": """make predictions for all the input
             data points""",
            "properties": {
                "prediction": {
                    "type": "string",
                    "description": "the column name for prediction",
                    "default": "predict"
                },
                "pred_contribs": {
                    "type": "boolean",
                    "description":
                    """
                    When this is True the output will be a matrix of size
                    (nsample, nfeats + 1) with each record indicating the
                    feature contributions (SHAP values) for that prediction.
                    The sum of all feature contributions is equal to the raw
                     untransformed margin value of the prediction. Note the
                      final column is the bias term.
                    """,
                    "default": False
                }
            },
            "required": [],
        }
        ui = {}
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        input_df = inputs[self.INPUT_PORT_NAME]
        bst_model = inputs[self.INPUT_PORT_MODEL_NAME]
        input_meta = self.get_input_meta()
        required_cols = input_meta[
            self.INPUT_PORT_MODEL_NAME]['train']
        required_cols = list(required_cols.keys())
        # required_cols.sort()
        predict_col = self.conf.get('prediction', 'predict')
        pred_contribs: bool = self.conf.get('pred_contribs', False)
        if isinstance(input_df, dask_cudf.DataFrame):
            # get the client
            client = dask.distributed.client.default_client()
            dtrain = xgb.dask.DaskDMatrix(client, input_df[required_cols])
            prediction = xgb.dask.predict(client,
                                          bst_model,
                                          dtrain,
                                          pred_contribs=pred_contribs)
            pred_df = dask_cudf.from_dask_dataframe(
                prediction.to_dask_dataframe())
            pred_df.index = input_df.index
            if not pred_contribs:
                input_df[predict_col] = pred_df
            else:
                input_df = pred_df
        else:
            infer_dmatrix = xgb.DMatrix(input_df[required_cols])
            if not pred_contribs:
                prediction = cudf.Series(bst_model.predict(infer_dmatrix),
                                         nan_as_null=False,
                                         index=input_df.index
                                         )
                input_df[predict_col] = prediction
            else:
                prediction = cudf.DataFrame(bst_model.predict(
                    infer_dmatrix, pred_contribs=pred_contribs),
                                            index=input_df.index)
                input_df = prediction
        return {self.OUTPUT_PORT_NAME: input_df}
