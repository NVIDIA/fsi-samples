from gquant.dataframe_flow.base_node import BaseNode
from gquant.dataframe_flow.portsSpecSchema import ConfSchema


class MaxNode(BaseNode):

    def init(self):
        super().init()
        self.INPUT_PORT_NAME = 'in'
        self.OUTPUT_PORT_NAME = 'out'
        cols_required = {"asset": "int64"}
        self.required = {
            self.INPUT_PORT_NAME: cols_required
        }

    def conf_schema(self):
        input_columns = self.get_input_columns()
        if self.INPUT_PORT_NAME in input_columns:
            col_from_inport = input_columns[self.INPUT_PORT_NAME]
            enums = [col for col in col_from_inport.keys()]
            json = {
                "title": "Maximum Value Node configure",
                "type": "object",
                "description": "Compute the maximum value of the key column",
                "properties": {
                    "column":  {
                        "type": "string",
                        "description": "column to calculate the maximum value",
                        "enum": enums
                    }
                },
                "required": ["column"],
            }
            ui = {
            }
            return ConfSchema(json=json, ui=ui)
        else:
            json = {
                "title": "Maximum Value Node configure",
                "type": "object",
                "description": "Compute the maximum value of the key column",
                "properties": {
                    "column":  {
                        "type": "string",
                        "description": "column to calculate the maximum value"
                    }
                },
                "required": ["column"],
            }
            ui = {
                "column": {"ui:widget": "text"}
            }
            return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        Compute the maximum value of the key column which is defined in the
        `column` of the node's conf

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        input_df = inputs[self.INPUT_PORT_NAME]
        max_column = self.conf['column']
        volume_df = input_df[[max_column,
                              "asset"]].groupby(["asset"]).max().reset_index()
        volume_df.columns = ['asset', max_column]
        return {self.OUTPUT_PORT_NAME: volume_df}

    def columns_setup(self):
        if 'column' in self.conf:
            retention = {self.conf['column']: "float64",
                         "asset": "int64"}
            return self.retention_columns_setup(retention)
        else:
            return self.retention_columns_setup({})
