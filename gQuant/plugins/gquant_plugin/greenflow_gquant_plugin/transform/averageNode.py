from greenflow.dataframe_flow import Node
from .._port_type_node import _PortTypesMixin
from greenflow.dataframe_flow.portsSpecSchema import ConfSchema


class AverageNode(_PortTypesMixin, Node):

    def init(self):
        _PortTypesMixin.init(self)
        self.INPUT_PORT_NAME = 'stock_in'
        self.OUTPUT_PORT_NAME = 'stock_out'

    def conf_schema(self):
        input_meta = self.get_input_meta()
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
        if self.INPUT_PORT_NAME in input_meta:
            col_from_inport = input_meta[self.INPUT_PORT_NAME]
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

    def meta_setup(self):
        required_col = {"asset": "int64"}
        if 'column' in self.conf:
            retention = {self.conf['column']: "float64",
                         "asset": "int64"}
            return _PortTypesMixin.retention_meta_setup(self,
                                                        retention,
                                                        required=required_col)
        else:
            retention = {"asset": "int64"}
            return _PortTypesMixin.retention_meta_setup(self,
                                                        retention,
                                                        required=required_col)

    def ports_setup(self):
        return _PortTypesMixin.ports_setup(self)
