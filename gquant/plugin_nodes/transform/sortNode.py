from gquant.dataframe_flow._port_type_node import _PortTypesMixin
from gquant.dataframe_flow.portsSpecSchema import ConfSchema
from gquant.dataframe_flow import Node


class SortNode(Node, _PortTypesMixin):

    def init(self):
        self.delayed_process = True
        _PortTypesMixin.init(self)
        cols_required = {}
        self.required = {
            self.INPUT_PORT_NAME: cols_required
        }

    def ports_setup(self):
        return _PortTypesMixin.ports_setup(self)

    def columns_setup(self):
        return _PortTypesMixin.columns_setup(self)

    def conf_schema(self):
        json = {
            "title": "Sort Column configure",
            "type": "object",
            "description": """Sort the input frames based on a
            list of columns, which are defined in the
            `keys` of the node's conf""",
            "properties": {
                "keys":  {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": """array of columns to sort"""
                }
            },
            "required": ["keys"],
        }
        ui = {
            "keys": {
                "items": {
                    "ui:widget": "text"
                }
            },
        }
        input_columns = self.get_input_columns()
        if self.INPUT_PORT_NAME in input_columns:
            col_from_inport = input_columns[self.INPUT_PORT_NAME]
            enums = [col for col in col_from_inport.keys()]
            json['properties']['keys']['items']['enum'] = enums
            ui = {}
            return ConfSchema(json=json, ui=ui)
        else:
            ui = {
                "column": {"ui:widget": "text"}
            }
            return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        Sort the input frames based on a list of columns, which are defined
        in the `keys` of the node's conf

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        input_df = inputs[self.INPUT_PORT_NAME]
        return {self.OUTPUT_PORT_NAME: input_df.sort_values(self.conf['keys'])}
