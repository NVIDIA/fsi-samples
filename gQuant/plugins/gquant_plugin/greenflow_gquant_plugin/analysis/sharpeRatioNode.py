from greenflow.dataframe_flow import Node, PortsSpecSchema
import math
import dask_cudf
from greenflow.dataframe_flow.portsSpecSchema import ConfSchema
from greenflow.dataframe_flow.metaSpec import MetaDataSchema
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin
from ..node_hdf_cache import NodeHDFCacheMixin


class SharpeRatioNode(TemplateNodeMixin, NodeHDFCacheMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.INPUT_PORT_NAME = 'stock_in'
        self.OUTPUT_PORT_NAME = 'sharpe_out'
        port_type = PortsSpecSchema.port_type
        port_inports = {
            self.INPUT_PORT_NAME: {
                port_type: [
                    "pandas.DataFrame", "cudf.DataFrame",
                    "dask_cudf.DataFrame", "dask.dataframe.DataFrame"
                ]
            },
        }
        port_outports = {
            self.OUTPUT_PORT_NAME: {
                port_type: ["builtins.float"]
            }
        }
        cols_required = {"strategy_returns": "float64"}
        retension = {}
        meta_inports = {
            self.INPUT_PORT_NAME: cols_required
        }
        meta_outports = {
            self.OUTPUT_PORT_NAME: {
                MetaDataSchema.META_OP: MetaDataSchema.META_OP_RETENTION,
                MetaDataSchema.META_DATA: retension
            }
        }
        self.template_ports_setup(
            in_ports=port_inports,
            out_ports=port_outports
        )
        self.template_meta_setup(
            in_ports=meta_inports,
            out_ports=meta_outports
        )

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
