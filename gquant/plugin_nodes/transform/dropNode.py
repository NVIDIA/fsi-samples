from gquant.dataframe_flow import Node
from gquant.dataframe_flow._port_type_node import _PortTypesMixin
from gquant.dataframe_flow.portsSpecSchema import ConfSchema


class DropNode(Node, _PortTypesMixin):

    def init(self):
        _PortTypesMixin.init(self)
        cols_required = {}
        self.required = {
            self.INPUT_PORT_NAME: cols_required
        }

    def columns_setup(self):
        if 'columns' in self.conf:
            dropped = {}
            for k in self.conf['columns']:
                dropped[k] = None
            return _PortTypesMixin.deletion_columns_setup(self,
                                                          dropped)
        else:
            return {self.OUTPUT_PORT_NAME: {}}

    def ports_setup(self):
        return _PortTypesMixin.ports_setup(self)

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
        input_columns = self.get_input_columns()
        if self.INPUT_PORT_NAME in input_columns:
            col_from_inport = input_columns[self.INPUT_PORT_NAME]
            enums = [col for col in col_from_inport.keys()]
            options = []
            for enum in enums:
                option = {
                          "type": "string",
                          "title": enum,
                          "enum": [enum]
                          }
                options.append(option)
            json['properties']['columns']['items']['anyOf'] = options
            ui = {}
            return ConfSchema(json=json, ui=ui)
        else:
            ui = {
                "column": {"ui:widget": "text"}
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
