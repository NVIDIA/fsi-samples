from .._port_type_node import _PortTypesMixin
from greenflow.dataframe_flow.portsSpecSchema import ConfSchema
from greenflow.dataframe_flow import Node


class SortNode(_PortTypesMixin, Node):

    def init(self):
        self.delayed_process = True
        _PortTypesMixin.init(self)

    def ports_setup(self):
        return _PortTypesMixin.ports_setup(self)

    def meta_setup(self):
        cols_required = {}
        return _PortTypesMixin.meta_setup(self,
                                          required=cols_required)

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
        input_meta = self.get_input_meta()
        if self.INPUT_PORT_NAME in input_meta:
            col_from_inport = input_meta[self.INPUT_PORT_NAME]
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
