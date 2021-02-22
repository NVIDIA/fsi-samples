from greenflow.dataframe_flow import Node
from .._port_type_node import _PortTypesMixin
from greenflow.dataframe_flow.portsSpecSchema import ConfSchema


class DropNode(_PortTypesMixin, Node):

    def init(self):
        _PortTypesMixin.init(self)

    def meta_setup(self):
        cols_required = {}
        if 'columns' in self.conf:
            dropped = {}
            for k in self.conf['columns']:
                dropped[k] = None
            return _PortTypesMixin.deletion_meta_setup(self,
                                                       dropped,
                                                       required=cols_required)
        else:
            return _PortTypesMixin.meta_setup(self, required=cols_required)

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
        input_meta = self.get_input_meta()
        if self.INPUT_PORT_NAME in input_meta:
            col_from_inport = input_meta[self.INPUT_PORT_NAME]
            enums = [col for col in col_from_inport.keys()]
            json['properties']['columns']['items']['enum'] = enums
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
