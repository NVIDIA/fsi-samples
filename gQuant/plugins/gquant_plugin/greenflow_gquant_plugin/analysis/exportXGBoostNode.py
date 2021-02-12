from greenflow.dataframe_flow import Node
from greenflow.dataframe_flow.portsSpecSchema import (ConfSchema, MetaData,
                                                   NodePorts, PortsSpecSchema)
from xgboost import Booster
from greenflow.dataframe_flow.util import get_file_path


class XGBoostExportNode(Node):

    def init(self):
        self.INPUT_PORT_NAME = 'model_in'
        self.OUTPUT_PORT_NAME = 'filename'

    def meta_setup(self):
        required = {self.INPUT_PORT_NAME: {}}
        input_meta = self.get_input_meta()
        output_cols = {
            self.OUTPUT_PORT_NAME: {}
        }
        if (self.INPUT_PORT_NAME in input_meta):
            output_cols = {
                self.OUTPUT_PORT_NAME: input_meta[self.INPUT_PORT_NAME]
            }
        metadata = MetaData(inports=required, outports=output_cols)
        return metadata

    def ports_setup(self):
        port_type = PortsSpecSchema.port_type
        input_ports = {
            self.INPUT_PORT_NAME: {
                port_type: [Booster, dict]
            }
        }
        output_ports = {
            self.OUTPUT_PORT_NAME: {
                port_type: str
            }
        }
        return NodePorts(inports=input_ports, outports=output_ports)

    def conf_schema(self):
        json = {
            "title": "XGBoost Export Configure",
            "type": "object",
            "description": """Export the xgboost model to a file
            """,
            "properties": {
                "path":  {
                    "type": "string",
                    "description": """The output filepath for the xgboost model"""
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
