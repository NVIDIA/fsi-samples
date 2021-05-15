from greenflow.dataframe_flow import Node, PortsSpecSchema
import mplfinance as mpf
from ipywidgets import Image
import dask_cudf
import io
import cudf
from greenflow.dataframe_flow.portsSpecSchema import ConfSchema
from greenflow.dataframe_flow.metaSpec import MetaDataSchema
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin
from ..node_hdf_cache import NodeHDFCacheMixin

__all__ = ['BarPlotNode']


class BarPlotNode(TemplateNodeMixin, NodeHDFCacheMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.INPUT_PORT_NAME = 'stock_in'
        self.OUTPUT_PORT_NAME = 'barplot'
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
                port_type: ["ipywidgets.Image"]
            }
        }
        cols_required = {"datetime": "datetime64[ns]",
                         "open": "float64",
                         "close": "float64",
                         "high": "float64",
                         "low": "float64",
                         "volume": "float64"}
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
            "title": "Bar Plot Node Configuration",
            "type": "object",
            "description": """Takes `datetime`, `open`, `close`, `high`,
            `volume` columns in the dataframe to plot the bqplot figure
            for financial bar data
            """,
            "properties": {
                "points":  {
                    "type": "number",
                    "description": "number of data points for the chart"
                },
                "label":  {
                    "type": "string",
                    "description": "label for the plot"
                },
            },
            "required": ["points"],
        }
        ui = {}
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        Takes `datetime`, `open`, `close`, `high`, `volume` columns in the
        dataframe to plot the bqplot figure for this stock.

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        bqplot.Figure
        """
        stock = inputs[self.INPUT_PORT_NAME]
        num_points = self.conf['points']
        stride = max(len(stock) // num_points, 1)
        buf = io.BytesIO()
        # Construct the marks
        if (isinstance(stock, cudf.DataFrame)
                or isinstance(stock, dask_cudf.DataFrame)):
            data_df = stock[[
                'datetime', 'open', 'high', 'low', 'close', 'volume'
            ]].iloc[::stride].to_pandas()
        else:
            data_df = stock[[
                'datetime', 'open', 'high', 'low', 'close', 'volume'
            ]].iloc[::stride]
        data_df.columns = [
            'Date', 'Open', 'High', 'Low', 'Close', 'Volume'
        ]
        data_df = data_df.set_index('Date')
        mpf.plot(data_df, type='candle', volume=True, savefig=buf)
        buf.seek(0)
        fig = Image(
            value=buf.read(),
            format='png',
            width=600,
            height=900,
        )
        return {self.OUTPUT_PORT_NAME: fig}
