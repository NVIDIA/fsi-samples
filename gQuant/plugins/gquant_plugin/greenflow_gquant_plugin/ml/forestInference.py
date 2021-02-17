from greenflow.dataframe_flow import Node
import cudf
import dask_cudf
from .._port_type_node import _PortTypesMixin
from greenflow.dataframe_flow.portsSpecSchema import (ConfSchema,
                                                   MetaData,
                                                   PortsSpecSchema, NodePorts)
import copy
from cuml import ForestInference
from greenflow.dataframe_flow.util import get_file_path


__all__ = ['ForestInferenceNode']


class ForestInferenceNode(_PortTypesMixin, Node):

    def init(self):
        _PortTypesMixin.init(self)
        self.delayed_process = True
        self.INPUT_PORT_NAME = 'data_in'
        self.INPUT_PORT_MODEL_NAME = 'model_file'
        self.OUTPUT_PORT_NAME = 'out'

    def ports_setup_from_types(self, types):
        port_type = PortsSpecSchema.port_type
        input_ports = {
            self.INPUT_PORT_NAME: {
                port_type: types
            },
            self.INPUT_PORT_MODEL_NAME: {
                port_type: str
            }
        }
        output_ports = {
            self.OUTPUT_PORT_NAME: {
                port_type: types
            }
        }
        input_connections = self.get_connected_inports()
        if self.INPUT_PORT_NAME in input_connections:
            determined_type = input_connections[self.INPUT_PORT_NAME]
            input_ports.update({self.INPUT_PORT_NAME:
                                {port_type: determined_type}})
            output_ports.update({self.OUTPUT_PORT_NAME:
                                {port_type: determined_type}})
            return NodePorts(inports=input_ports,
                             outports=output_ports)
        else:
            return NodePorts(inports=input_ports, outports=output_ports)

    def meta_setup(self):
        input_meta = self.get_input_meta()
        required = {self.INPUT_PORT_NAME: {},
                    self.INPUT_PORT_MODEL_NAME: {}}
        predict = self.conf.get('prediction', 'predict')
        output_cols = {
            self.OUTPUT_PORT_NAME: {predict: None}
        }
        if (self.INPUT_PORT_NAME in input_meta
                and self.INPUT_PORT_MODEL_NAME in input_meta):
            col_from_inport = input_meta[self.INPUT_PORT_NAME]
            if 'train' in input_meta[self.INPUT_PORT_MODEL_NAME]:
                required_cols = input_meta[self.INPUT_PORT_MODEL_NAME]['train']
            else:
                required_cols = {}
            predict = self.conf.get('prediction', 'predict')
            col_from_inport[predict] = None  # the type is not determined
            required = {self.INPUT_PORT_NAME: required_cols,
                        self.INPUT_PORT_MODEL_NAME: {}}
            output_cols = {
                self.OUTPUT_PORT_NAME: col_from_inport,
            }
            metadata = MetaData(inports=required, outports=output_cols)
            return metadata
        elif (self.INPUT_PORT_NAME not in input_meta and
              self.INPUT_PORT_MODEL_NAME in input_meta):
            if 'train' in input_meta[self.INPUT_PORT_MODEL_NAME]:
                required_cols = input_meta[self.INPUT_PORT_MODEL_NAME]['train']
            else:
                required_cols = {}
            predict = self.conf.get('prediction', 'predict')
            col_from_inport = copy.copy(required_cols)
            col_from_inport[predict] = None
            # the type is not determined
            required = {self.INPUT_PORT_NAME: required_cols,
                        self.INPUT_PORT_MODEL_NAME: {}}
            output_cols = {
                self.OUTPUT_PORT_NAME: col_from_inport
            }
            metadata = MetaData(inports=required, outports=output_cols)
            return metadata
        elif (self.INPUT_PORT_NAME in input_meta and
              self.INPUT_PORT_MODEL_NAME not in input_meta):
            cols_required = {}
            col_from_inport = input_meta[self.INPUT_PORT_NAME]
            enums = [col for col in col_from_inport.keys()]
            if 'columns' in self.conf:
                if self.conf.get('include', True):
                    included_colums = self.conf['columns']
                else:
                    included_colums = [col for col in enums
                                       if col not in self.conf['columns']]
                for col in included_colums:
                    if col in col_from_inport:
                        cols_required[col] = col_from_inport[col]
                    else:
                        cols_required[col] = None
            predict = self.conf.get('prediction', 'predict')
            col_from_inport[predict] = None  # the type is not determined
            required = {
                 self.INPUT_PORT_NAME: cols_required,
            }
            output_cols = {
                self.OUTPUT_PORT_NAME: col_from_inport
            }
            metadata = MetaData(inports=required, outports=output_cols)
            return metadata
        elif (self.INPUT_PORT_NAME not in input_meta and
              self.INPUT_PORT_MODEL_NAME not in input_meta):
            if 'columns' in self.conf:
                if self.conf.get('include', True):
                    included_colums = self.conf['columns']
                    cols_required = {}
                    for col in included_colums:
                        cols_required[col] = None
                    required = {
                        self.INPUT_PORT_NAME: cols_required,
                    }
        metadata = MetaData(inports=required, outports=output_cols)
        return metadata

    def ports_setup(self):
        types = [cudf.DataFrame,
                 dask_cudf.DataFrame]
        return self.ports_setup_from_types(types)

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
