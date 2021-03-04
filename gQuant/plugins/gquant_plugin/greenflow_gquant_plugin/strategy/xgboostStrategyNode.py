from greenflow.dataframe_flow import Node
import cudf
import dask_cudf
from .._port_type_node import _PortTypesMixin
from greenflow.dataframe_flow.portsSpecSchema import ConfSchema


__all__ = ['XGBoostStrategyNode']


class XGBoostStrategyNode(_PortTypesMixin, Node):
    """
    This is the Node used to compute trading signal from XGBoost Strategy.

    """

    def init(self):
        _PortTypesMixin.init(self)
        self.INPUT_PORT_NAME = 'stock_in'
        self.OUTPUT_PORT_NAME = 'stock_out'

    def meta_setup(self):
        cols_required = {'predict': None, "asset": "int64"}
        addition = {}
        addition['signal'] = 'float64'
        return _PortTypesMixin.addition_meta_setup(self, addition,
                                                   cols_required)

    def ports_setup(self):
        types = [cudf.DataFrame,
                 dask_cudf.DataFrame]
        return _PortTypesMixin.ports_setup_from_types(self, types)

    def conf_schema(self):
        json = {
            "title": "XGBoost Node configure",
            "type": "object",
            "description": """convert the predicted next day return as trading actions
            """,
            "properties": {
            },
        }
        ui = {
        }
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        input_df = inputs[self.INPUT_PORT_NAME]
        # convert the signal to trading action
        # 1 is buy and -1 is sell
        # It predicts the tomorrow's return (shift -1)
        # We shift 1 for trading actions so that it acts on the second day
        input_df['signal'] = ((
            input_df['predict'] >= 0).astype('float') * 2 - 1).shift(1)
        # remove the bad datapints
        input_df = input_df.dropna()
        return {self.OUTPUT_PORT_NAME: input_df}
