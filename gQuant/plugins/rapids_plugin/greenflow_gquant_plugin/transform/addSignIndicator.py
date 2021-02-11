from greenflow.dataframe_flow import Node
from greenflow.dataframe_flow.portsSpecSchema import ConfSchema
from .._port_type_node import _PortTypesMixin

__all__ = ["AddSignIndicatorNode"]


class AddSignIndicatorNode(_PortTypesMixin, Node):

    def init(self):
        _PortTypesMixin.init(self)
        self.INPUT_PORT_NAME = 'in'
        self.OUTPUT_PORT_NAME = 'out'

    def ports_setup(self):
        return _PortTypesMixin.ports_setup(self)

    def conf_schema(self):
        json = {
            "title": "Add Sign Indicator configure",
            "type": "object",
            "description": """If the number is bigger than zero,
            the sign is 1, otherwise the sign is 0
            """,
            "properties": {
                "column":  {
                    "type": "string",
                    "description": """the column that is used to calcuate
                    sign"""
                },
                "sign":  {
                    "type": "string",
                    "description": "the sign column name",
                    "default": "sign"
                }
            },
            "required": ["column"],
        }
        ui = {
        }
        input_meta = self.get_input_meta()
        if self.INPUT_PORT_NAME in input_meta:
            col_from_inport = input_meta[self.INPUT_PORT_NAME]
            enums = [col for col in col_from_inport.keys()]
            json['properties']['column']['enum'] = enums
            return ConfSchema(json=json, ui=ui)
        else:
            return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        Rename the column name in the datafame from `old` to `new` defined in
        the node's conf

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        input_df = inputs[self.INPUT_PORT_NAME]
        name = self.conf.get('sign', 'sign')
        input_df[name] = (input_df[self.conf['column']] > 0).astype('int64')
        return {self.OUTPUT_PORT_NAME: input_df}

    def meta_setup(self):
        name = self.conf.get('sign', 'sign')
        addition = {name: "int64"}
        cols_required = {}
        return _PortTypesMixin.addition_meta_setup(self,
                                                   addition,
                                                   required=cols_required)
