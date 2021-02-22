from greenflow.dataframe_flow import Node
from .._port_type_node import _PortTypesMixin
from greenflow.dataframe_flow.portsSpecSchema import ConfSchema


class SimpleBackTestNode(Node, _PortTypesMixin):

    def init(self):
        _PortTypesMixin.init(self)
        self.INPUT_PORT_NAME = 'bardata_in'
        self.OUTPUT_PORT_NAME = 'backtest_out'

    def meta_setup(self):
        cols_required = {"signal": "float64",
                         "returns": "float64"}
        addition = {"strategy_returns": "float64"}
        return _PortTypesMixin.addition_meta_setup(self, addition,
                                                   required=cols_required)

    def ports_setup(self):
        return _PortTypesMixin.ports_setup(self)

    def conf_schema(self):
        json = {
            "title": "Backtest configure",
            "type": "object",
            "description": """compute the `strategy_returns` by assuming invest
             `signal` amount of dollars for each of the time step.""",
        }
        ui = {
        }
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        compute the `strategy_returns` by assuming invest `signal` amount of
        dollars for each of the time step.

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        input_df = inputs[self.INPUT_PORT_NAME]
        input_df['strategy_returns'] = input_df['signal'] * input_df['returns']
        return {self.OUTPUT_PORT_NAME: input_df}
