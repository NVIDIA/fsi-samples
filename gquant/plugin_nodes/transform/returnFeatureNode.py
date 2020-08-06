from gquant.dataframe_flow import Node
from gquant.dataframe_flow._port_type_node import _PortTypesMixin
from gquant.dataframe_flow.portsSpecSchema import ConfSchema


class ReturnFeatureNode(Node, _PortTypesMixin):

    def init(self):
        self.delayed_process = True
        _PortTypesMixin.init(self)
        self.INPUT_PORT_NAME = 'stock_in'
        self.OUTPUT_PORT_NAME = 'stock_out'
        cols_required = {"close": "float64",
                         "asset": "int64"}
        self.required = {
            self.INPUT_PORT_NAME: cols_required
        }

    def columns_setup(self):
        addition = {"returns": "float64"}
        return _PortTypesMixin.addition_columns_setup(self,
                                                      addition)

    def conf_schema(self):
        json = {
            "title": "Add Returen Feature Node configure",
            "type": "object",
            "description": """Add the rate of of return column based
            on the `close` price for each of the asset in the dataframe.
            """,
        }
        ui = {
        }
        return ConfSchema(json=json, ui=ui)

    def ports_setup(self):
        return _PortTypesMixin.ports_setup(self)

    def process(self, inputs):
        """
        Add the rate of of return column based on the `close` price for each
        of the asset in the dataframe. The result column is named as `returns`

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        tmp_col = "ae699380a8834957b3a8b7ad60192dd7"
        input_df = inputs[self.INPUT_PORT_NAME]
        shifted = input_df['close'].shift(1)
        input_df['returns'] = (input_df['close'] - shifted) / shifted
        input_df['returns'] = input_df['returns'].fillna(0.0)
        input_df[tmp_col] = (input_df['asset'] -
                             input_df['asset'].shift(1)).fillna(1)
        input_df[tmp_col] = (input_df[tmp_col] != 0).astype('int32')
        input_df[tmp_col][input_df[tmp_col] == 1] = None
        return {self.OUTPUT_PORT_NAME: input_df.dropna(
            subset=[tmp_col]).drop(tmp_col, axis=1)}
