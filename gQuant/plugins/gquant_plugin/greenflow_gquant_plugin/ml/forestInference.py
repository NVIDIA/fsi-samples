from cuml import ForestInference
from greenflow.dataframe_flow import Node
from greenflow.dataframe_flow.portsSpecSchema import (ConfSchema,
                                                      PortsSpecSchema)
from greenflow.dataframe_flow.metaSpec import MetaDataSchema
from greenflow.dataframe_flow.util import get_file_path
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin
from ..node_hdf_cache import NodeHDFCacheMixin

__all__ = ['ForestInferenceNode']


class ForestInferenceNode(TemplateNodeMixin, NodeHDFCacheMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.delayed_process = True
        self.INPUT_PORT_NAME = 'data_in'
        self.INPUT_PORT_MODEL_NAME = 'model_file'
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
                port_type: ['builtins.str']
            }

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
        meta_outports = {
            self.OUTPUT_PORT_NAME: {
                MetaDataSchema.META_OP: MetaDataSchema.META_OP_ADDITION,
                MetaDataSchema.META_REF_INPUT: self.INPUT_PORT_NAME,
                MetaDataSchema.META_DATA: {predict: None}
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
        meta_inports = self.template_meta_setup().inports
        if self.INPUT_PORT_MODEL_NAME in input_meta:
            if 'train' in input_meta[self.INPUT_PORT_MODEL_NAME]:
                required_cols = input_meta[self.INPUT_PORT_MODEL_NAME]['train']
            else:
                required_cols = {}
            meta_inports[self.INPUT_PORT_NAME] = required_cols
        else:
            if self.INPUT_PORT_NAME in input_meta:
                col_from_inport = input_meta[self.INPUT_PORT_NAME]
            else:
                col_from_inport = {}
            enums = [col for col in col_from_inport.keys()]
            if 'columns' in self.conf:
                if self.conf.get('include', True):
                    included_colums = self.conf['columns']
                else:
                    included_colums = [col for col in enums
                                       if col not in self.conf['columns']]
                for col in included_colums:
                    if col in col_from_inport:
                        meta_inports[
                            self.INPUT_PORT_NAME][col] = col_from_inport[col]
                    else:
                        meta_inports[self.INPUT_PORT_NAME][col] = None
        self.template_meta_setup(
            in_ports=meta_inports,
            out_ports=None
        )

    def conf_schema(self):
        json = {
            "title": "Forest Inferencing Node",
            "type": "object",
            "description": """ForestInference provides GPU-accelerated inference
            (prediction) for random forest and boosted decision tree models.
            This module does not support training models. Rather, users should
            train a model in another package and save it in a
            treelite-compatible format. (See https://github.com/dmlc/treelite)
            Currently, LightGBM, XGBoost and SKLearn GBDT and random forest
            models are supported.""",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                    "description": """columns in the input dataframe that
        are considered as input features or not depending on `include` flag."""
                },
                "include":  {
                    "type": "boolean",
                    "description": """if set true, the `columns` are treated as
                    input features if false, all dataframe columns are input
                    features except the `columns`""",
                    "default": True
                },
                "file": {
                    "type": "string",
                    "description": """The saved model file"""
                },
                "prediction":  {
                    "type": "string",
                    "description": "the column name for prediction",
                    "default": "predict"
                },
                "model_type":  {
                    "type": "string",
                    "description": """Format of the saved treelite model to be
                        load""",
                    "enum": ["xgboost", "lightgbm"],
                    "default": "xgboost"
                },
            },
            "required": ['file'],
        }
        ui = {
            "file": {"ui:widget": "FileSelector"},
        }
        input_meta = self.get_input_meta()
        if self.INPUT_PORT_NAME in input_meta:
            col_from_inport = input_meta[self.INPUT_PORT_NAME]
            enums = [col for col in col_from_inport.keys()]
            json['properties']['columns']['items']['enum'] = enums
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        input_meta = self.get_input_meta()
        predict_col = self.conf.get('prediction', 'predict')
        data_df = inputs[self.INPUT_PORT_NAME]

        if self.INPUT_PORT_MODEL_NAME in input_meta:
            # use external information instead of conf
            filename = get_file_path(inputs[self.INPUT_PORT_MODEL_NAME])
            train_cols = input_meta[self.INPUT_PORT_MODEL_NAME]['train']
            train_cols = list(train_cols.keys())
        else:
            # use the conf information
            filename = get_file_path(self.conf['file'])
            if 'columns' in self.conf:
                if self.conf.get('include', True):
                    train_cols = self.conf['columns']
                else:
                    train_cols = [col for col in data_df.columns
                                  if col not in self.conf['columns']]
        # train_cols.sort()
        fm = ForestInference.load(filename,
                                  model_type=self.conf.get("model_type",
                                                           "xgboost"))
        prediction = fm.predict(data_df[train_cols])
        prediction.index = data_df.index
        data_df[predict_col] = prediction
        return {self.OUTPUT_PORT_NAME: data_df}
