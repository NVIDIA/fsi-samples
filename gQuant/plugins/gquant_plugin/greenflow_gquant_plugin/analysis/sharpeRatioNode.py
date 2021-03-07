from greenflow.dataframe_flow import Node, PortsSpecSchema
import math
import dask_cudf
from greenflow.dataframe_flow.portsSpecSchema import ConfSchema
from .._port_type_node import _PortTypesMixin


class SharpeRatioNode(_PortTypesMixin, Node):

    def init(self):
        _PortTypesMixin.init(self)
        self.INPUT_PORT_NAME = 'stock_in'
        self.OUTPUT_PORT_NAME = 'sharpe_out'
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
                port_type: ["builtins.float"]
            }
        }
        cols_required = {"strategy_returns": "float64"}
        retension = {}
        self.meta_inports = {
            self.INPUT_PORT_NAME: cols_required
        }
        self.meta_outports = {
            self.OUTPUT_PORT_NAME: {
                self.META_OP: self.META_OP_RETENTION,
                self.META_DATA: retension
            }
        }

    def ports_setup(self):
        return _PortTypesMixin.ports_setup(self)

    def meta_setup(self):
        return _PortTypesMixin.meta_setup(self)

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
