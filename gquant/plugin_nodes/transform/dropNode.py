from gquant.dataframe_flow.base_node import BaseNode
from gquant.dataframe_flow.portsSpecSchema import ConfSchema


class DropNode(BaseNode):

    def init(self):
        super().init()
        cols_required = {}
        self.required = {
            self.INPUT_PORT_NAME: cols_required
        }

    def columns_setup(self):
        dropped = {}
        for k in self.conf['columns']:
            dropped[k] = None
        return self.deletion_columns_setup(dropped)

    def conf_schema(self):
        json = {
            "title": "Drop Column configure",
            "type": "object",
            "description": """Drop a few columns from the dataframe""",
            "properties": {
                "columns":  {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": """array of columns to be droped"""
                }
            },
            "required": ["columns"],
        }
        ui = {
            "columns": {
                "items": {
                    "ui:widget": "text"
                }
            },

        }
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        Drop a few columns from the dataframe that are defined in the `columns`
        in the nodes' conf

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        input_df = inputs[self.INPUT_PORT_NAME]
        column_names = self.conf['columns']
        return {self.OUTPUT_PORT_NAME: input_df.drop(column_names, axis=1)}
