from greenflow.dataframe_flow import Node
from greenflow.dataframe_flow.portsSpecSchema import (ConfSchema,
                                                      PortsSpecSchema)
from greenflow.dataframe_flow.util import get_file_path
from .._port_type_node import _PortTypesMixin


class XGBoostExportNode(_PortTypesMixin, Node):

    def init(self):
        _PortTypesMixin.init(self)
        self.INPUT_PORT_NAME = 'model_in'
        self.OUTPUT_PORT_NAME = 'filename'
        port_type = PortsSpecSchema.port_type
        self.port_inports = {
            self.INPUT_PORT_NAME: {
                port_type: ["xgboost.Booster", "builtins.dict"]
            }
        }
        self.port_outports = {
            self.OUTPUT_PORT_NAME: {
                port_type: ["builtins.str"]
            }
        }
        cols_required = {}
        addition = {}
        self.meta_inports = {
            self.INPUT_PORT_NAME: cols_required
        }
        self.meta_outports = {
            self.OUTPUT_PORT_NAME: {
                self.META_OP: self.META_OP_ADDITION,
                self.META_REF_INPUT: self.INPUT_PORT_NAME,
                self.META_DATA: addition
            }
        }

    def ports_setup(self):
        return _PortTypesMixin.ports_setup(self)

    def meta_setup(self):
        return _PortTypesMixin.meta_setup(self)

    def conf_schema(self):
        json = {
            "title": "XGBoost Export Configure",
            "type": "object",
            "description": """Export the xgboost model to a file
            """,
            "properties": {
                "path": {
                    "type": "string",
                    "description":
                    """The output filepath for the xgboost
                     model"""
                }
            },
            "required": ["path"],
        }
        ui = {}
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        dump the model into the file
        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """

        model = inputs[self.INPUT_PORT_NAME]
        if isinstance(model,  dict):
            model = model['booster']
        pathname = get_file_path(self.conf['path'])
        model.save_model(pathname)
        return {self.OUTPUT_PORT_NAME: pathname}
