from greenflow.dataframe_flow import Node
from greenflow.dataframe_flow.portsSpecSchema import (ConfSchema,
                                                      PortsSpecSchema)
from greenflow.dataframe_flow.metaSpec import MetaDataSchema
from greenflow.dataframe_flow.util import get_file_path
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin
from ..node_hdf_cache import NodeHDFCacheMixin


class XGBoostExportNode(TemplateNodeMixin, NodeHDFCacheMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.INPUT_PORT_NAME = 'model_in'
        self.OUTPUT_PORT_NAME = 'filename'
        port_type = PortsSpecSchema.port_type
        port_inports = {
            self.INPUT_PORT_NAME: {
                port_type: ["xgboost.Booster", "builtins.dict"]
            }
        }
        port_outports = {
            self.OUTPUT_PORT_NAME: {
                port_type: ["builtins.str"]
            }
        }
        cols_required = {}
        addition = {}
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
