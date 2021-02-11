from greenflow.dataframe_flow import Node
from .._port_type_node import _PortTypesMixin
from greenflow.dataframe_flow.portsSpecSchema import ConfSchema


class AssetIndicatorNode(_PortTypesMixin, Node):

    def init(self):
        self.delayed_process = True
        _PortTypesMixin.init(self)
        self.INPUT_PORT_NAME = 'stock_in'
        self.OUTPUT_PORT_NAME = 'stock_out'

    def meta_setup(self):
        cols_required = {"asset": "int64"}
        return _PortTypesMixin.addition_meta_setup(self,
                                                   {"indicator": "int32"},
                                                   required=cols_required)

    def ports_setup(self):
        return _PortTypesMixin.ports_setup(self)

    def conf_schema(self):
        json = {
            "title": "Asset indicator configure",
            "type": "object",
            "description": """Add the indicator column in the dataframe which
             set 1 at the beginning of the each of the assets, assuming the
             rows are sorted so same asset are grouped together""",
        }
        ui = {
        }
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        Add the indicator column in the dataframe which set 1 at the beginning
        of the each of the assets, assuming the rows are sorted so same asset
        are grouped together

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """

        input_df = inputs[self.INPUT_PORT_NAME]
        input_df['indicator'] = (input_df['asset'] -
                                 input_df['asset'].shift(1)).fillna(1)
        input_df['indicator'] = (input_df['indicator'] != 0).astype('int32')
        return {self.OUTPUT_PORT_NAME: input_df}
