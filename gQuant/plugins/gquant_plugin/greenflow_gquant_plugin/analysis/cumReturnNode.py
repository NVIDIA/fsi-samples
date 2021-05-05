from greenflow.dataframe_flow import Node, PortsSpecSchema
import matplotlib as mpl
import matplotlib.pyplot as plt
from dask.dataframe import DataFrame as DaskDataFrame
import cudf
from greenflow.dataframe_flow.portsSpecSchema import ConfSchema
from greenflow.dataframe_flow.metaSpec import MetaDataSchema
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin
from ..node_hdf_cache import NodeHDFCacheMixin


class CumReturnNode(TemplateNodeMixin, NodeHDFCacheMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.INPUT_PORT_NAME = 'in'
        self.OUTPUT_PORT_NAME = 'cum_return'
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
                port_type: ["matplotlib.figure.Figure"]
            }
        }
        cols_required = {"datetime": "datetime64[ns]",
                         "strategy_returns": "float64"}
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
            "title": "Cumulative Return Configuration",
            "type": "object",
            "description": """Plot the P & L graph from the `strategy_returns` column.
            """,
            "properties": {
                "points":  {
                    "type": "number",
                    "description": "number of data points for the chart"
                },
                "label":  {
                    "type": "string",
                    "description": "Label for the line plot"
                },
            },
            "required": ["points"],
        }
        ui = {
        }
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        Plot the P & L graph from the `strategy_returns` column.
        `label` in the `conf` defines the stock symbol name

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        Figure

        """
        input_df = inputs[self.INPUT_PORT_NAME]
        if isinstance(input_df,  DaskDataFrame):
            input_df = input_df.compute()  # get the computed value
        label = 'stock'
        if 'label' in self.conf:
            label = self.conf['label']
        num_points = self.conf['points']
        stride = max(len(input_df) // num_points, 1)
        backend_ = mpl.get_backend()
        mpl.use("Agg")  # Prevent showing stuff

        f = plt.figure()
        if (isinstance(input_df,
                       cudf.DataFrame) or isinstance(input_df,
                                                     DaskDataFrame)):
            plt.plot(input_df['datetime'][::stride].to_array(), (input_df[
                             'strategy_returns'].cumsum())[
                                 ::stride].to_array(), 'b', label=label)
        else:
            plt.plot(input_df['datetime'][::stride],
                     (input_df['strategy_returns'].cumsum())[::stride],
                     'b',
                     label=label)
        plt.xlabel("Time")
        plt.ylabel("Cumulative return")
        plt.grid(True)
        mpl.use(backend_)
        return {self.OUTPUT_PORT_NAME: f}
