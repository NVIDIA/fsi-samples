from greenflow.dataframe_flow import Node, PortsSpecSchema
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
        port_type = PortsSpecSchema.port_type
        self.port_inports = {
            self.INPUT_PORT_NAME: {
                port_type: [
                    "pandas.DataFrame", "cudf.DataFrame",
                    "dask_cudf.DataFrame", "dask.dataframe.DataFrame"
                ]
            },
        }
        self.port_outports = {
            self.OUTPUT_PORT_NAME: {
                port_type: "${port:stock_in}"
            }
        }
        cols_required = {'predict': None, "asset": "int64"}
        addition = {}
        addition['signal'] = 'float64'
        self.meta_inports = {
            self.INPUT_PORT_NAME: cols_required
        }
        self.meta_outports = {
            self.OUTPUT_PORT_NAME: {
                self.META_OP: self.META_OP_ADDITION,
                self.META_REF_INPUT: self.INPUT_PORT_NAME,
                self.META_DATA: addition
            }
        }

    def meta_setup(self):
        return _PortTypesMixin.meta_setup(self)

    def ports_setup(self):
        return _PortTypesMixin.ports_setup(self)

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
