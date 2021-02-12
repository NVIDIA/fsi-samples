from greenflow.dataframe_flow.portsSpecSchema import ConfSchema
from greenflow.dataframe_flow import Node
from .._port_type_node import _PortTypesMixin


class MinNode(_PortTypesMixin, Node):

    def init(self):
        _PortTypesMixin.init(self)
        self.INPUT_PORT_NAME = 'in'
        self.OUTPUT_PORT_NAME = 'out'

    def ports_setup(self):
        return _PortTypesMixin.ports_setup(self)

    def conf_schema(self):
        json = {
            "title": "Minimum Value Node configure",
            "type": "object",
            "description": "Compute the minimum value of the key column",
            "properties": {
                "column":  {
                    "type": "string",
                    "description": "column to calculate the minimum value"
                }
            },
            "required": ["column"],
        }
        input_meta = self.get_input_meta()
        if self.INPUT_PORT_NAME in input_meta:
            col_from_inport = input_meta[self.INPUT_PORT_NAME]
            enums = [col for col in col_from_inport.keys()]
            json['properties']['column']['enum'] = enums
            ui = {
            }
            return ConfSchema(json=json, ui=ui)
        else:
            ui = {
                "column": {"ui:widget": "text"}
            }
            return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        Compute the minium value of the key column which is defined in the
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
        min_column = self.conf['column']
        volume_df = input_df[[min_column,
                              "asset"]].groupby(["asset"]).min().reset_index()
        volume_df.columns = ['asset', min_column]
        return {self.OUTPUT_PORT_NAME: volume_df}

    def meta_setup(self):
        cols_required = {"asset": "int64"}
        if 'column' in self.conf:
            retention = {self.conf['column']: "float64",
                         "asset": "int64"}
            return _PortTypesMixin.retention_meta_setup(self,
                                                        retention,
                                                        required=cols_required)
        else:
            retention = {"asset": "int64"}
            return _PortTypesMixin.retention_meta_setup(self,
                                                        retention,
                                                        required=cols_required)
