from gquant.dataframe_flow.base_node import BaseNode
from gquant.dataframe_flow.portsSpecSchema import ConfSchema


class AverageNode(BaseNode):

    def init(self):
        super().init()
        self.INPUT_PORT_NAME = 'stock_in'
        self.OUTPUT_PORT_NAME = 'stock_out'
        required = {"asset": "int64"}
        self.required = {self.INPUT_PORT_NAME: required}

    def conf_schema(self):
        json = {
            "title": "Asset Average Configure",
            "type": "object",
            "description": """Compute the average value of the key column
            which is defined in the configuration
            """,
            "properties": {
                "column":  {
                    "type": "string",
                    "description": """the column name in the dataframe
                    to average"""
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
        Compute the average value of the key column which is defined in the
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
        average_column = self.conf['column']
        volume_df = input_df[[average_column, "asset"]] \
            .groupby(["asset"]).mean().reset_index()
        volume_df.columns = ['asset', average_column]
        return {self.OUTPUT_PORT_NAME: volume_df}

    def columns_setup(self):
        retention = {self.conf['column']: "float64",
                     "asset": "int64"}
        return self.retention_columns_setup(retention)
