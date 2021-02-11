from greenflow.dataframe_flow import Node
import math
import dask_cudf
from greenflow.dataframe_flow.portsSpecSchema import ConfSchema, MetaData
from .._port_type_node import _PortTypesMixin


class SharpeRatioNode(Node, _PortTypesMixin):

    def init(self):
        self.INPUT_PORT_NAME = 'stock_in'
        self.OUTPUT_PORT_NAME = 'sharpe_out'

    def meta_setup(self):
        cols_required = {"strategy_returns": "float64"}
        required = {
            self.INPUT_PORT_NAME: cols_required
        }
        metadata = MetaData(inports=required,
                            outports={self.OUTPUT_PORT_NAME: {}})
        return metadata

    def ports_setup(self):
        return _PortTypesMixin.ports_setup_different_output_type(self,
                                                                 float)

    def conf_schema(self):
        json = {
            "title": "Calculate Sharpe Ratio configure",
            "type": "object",
            "description": """Compute the yearly Sharpe Ratio from the
            input dataframe `strategy_returns` column. Assume it is
            daily return. Asumes 252 trading days per year
            """,
        }
        ui = {
        }
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        Compute the yearly Sharpe Ratio from the input dataframe
        `strategy_returns` column. Assume it is daily return. Asumes
        252 trading days per year.

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        float
            the sharpe ratio
        """

        input_df = inputs[self.INPUT_PORT_NAME]
        if isinstance(input_df,  dask_cudf.DataFrame):
            input_df = input_df.compute()  # get the computed value
        daily_mean = input_df['strategy_returns'].mean()
        daily_std = input_df['strategy_returns'].std()
        return {self.OUTPUT_PORT_NAME: float(
            daily_mean / daily_std * math.sqrt(252))}
