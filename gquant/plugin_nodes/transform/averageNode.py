from gquant.dataframe_flow import Node
from gquant.dataframe_flow._port_type_node import _PortTypesMixin
from gquant.dataframe_flow.portsSpecSchema import ConfSchema


class AverageNode(Node, _PortTypesMixin):

    def init(self):
        _PortTypesMixin.init(self)
        self.INPUT_PORT_NAME = 'stock_in'
        self.OUTPUT_PORT_NAME = 'stock_out'
        required = {"asset": "int64"}
        self.required = {self.INPUT_PORT_NAME: required}

    def conf_schema(self):
        input_columns = self.get_input_columns()
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
        if self.INPUT_PORT_NAME in input_columns:
            col_from_inport = input_columns[self.INPUT_PORT_NAME]
            enums = [col for col in col_from_inport.keys()]
            json['properties']['column']['enum'] = enums
            ui = {}
            return ConfSchema(json=json, ui=ui)
        else:
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
        if 'column' in self.conf:
            retention = {self.conf['column']: "float64",
                         "asset": "int64"}
            return _PortTypesMixin.retention_columns_setup(self,
                                                           retention)
        else:
            retention = {"asset": "int64"}
            return _PortTypesMixin.retention_columns_setup(self,
                                                           retention)

    def ports_setup(self):
        return _PortTypesMixin.ports_setup(self)
