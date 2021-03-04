from greenflow.dataframe_flow import Node
# from bqplot import Axis, LinearScale, DateScale, Figure, OHLC, Bars, Tooltip
import mplfinance as mpf
from ipywidgets import Image
import dask_cudf
import io
import cudf
from greenflow.dataframe_flow.portsSpecSchema import ConfSchema, MetaData
from .._port_type_node import _PortTypesMixin


class BarPlotNode(Node):

    def init(self):
        self.INPUT_PORT_NAME = 'stock_in'
        self.OUTPUT_PORT_NAME = 'barplot'

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
        ui = {
        }
        return ConfSchema(json=json, ui=ui)

    def ports_setup(self):
        return _PortTypesMixin.ports_setup_different_output_type(self,
                                                                 Image)

    def meta_setup(self):
        cols_required = {"datetime": "datetime64[ns]",
                         "open": "float64",
                         "close": "float64",
                         "high": "float64",
                         "low": "float64",
                         "volume": "float64"}
        required = {
            self.INPUT_PORT_NAME: cols_required
        }
        metadata = MetaData(inports=required,
                            outports={self.OUTPUT_PORT_NAME: {}})
        return metadata

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
