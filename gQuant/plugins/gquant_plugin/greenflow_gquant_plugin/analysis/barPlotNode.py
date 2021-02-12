from greenflow.dataframe_flow import Node
from bqplot import Axis, LinearScale, DateScale, Figure, OHLC, Bars, Tooltip
import cupy as cp
import cudf
import dask_cudf
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
                                                                 Figure)

    def meta_setup(self):
        cols_required = {"datetime": "date",
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
        label = 'stock'
        if 'label' in self.conf:
            label = self.conf['label']
        sc = LinearScale()
        sc2 = LinearScale()
        dt_scale = DateScale()
        ax_x = Axis(label='Date', scale=dt_scale)
        ax_y = Axis(label='Price', scale=sc, orientation='vertical',
                    tick_format='0.0f')
        # Construct the marks
        if (isinstance(stock,
                       cudf.DataFrame) or isinstance(stock,
                                                     dask_cudf.DataFrame)):
            ohlc = OHLC(x=stock['datetime'][::stride].to_array(),
                        y=cp.asnumpy(stock[['open',
                                            'high',
                                            'low',
                                            'close']].values[::stride, :]),
                        marker='candle', scales={'x': dt_scale, 'y': sc},
                        format='ohlc', stroke='blue',
                        display_legend=True, labels=[label])
            bar = Bars(x=stock['datetime'][::stride].to_array(),
                       y=stock['volume'][::stride].to_array(),
                       scales={'x': dt_scale, 'y': sc2},
                       padding=0.2)
        else:
            ohlc = OHLC(x=stock['datetime'][::stride],
                        y=stock[['open',
                                 'high',
                                 'low', 'close']].values[::stride, :],
                        marker='candle', scales={'x': dt_scale, 'y': sc},
                        format='ohlc', stroke='blue',
                        display_legend=True, labels=[label])
            bar = Bars(x=stock['datetime'][::stride],
                       y=stock['volume'][::stride],
                       scales={'x': dt_scale, 'y': sc2},
                       padding=0.2)
        def_tt = Tooltip(fields=['x', 'y'], formats=['%Y-%m-%d', '.2f'])
        bar.tooltip = def_tt
        bar.interactions = {
            'legend_hover': 'highlight_axes',
            'hover': 'tooltip',
            'click': 'select',
         }
        sc.min = stock['close'].min() - 0.3 * \
            (stock['close'].max() - stock['close'].min())
        sc.max = stock['close'].max()
        sc2.max = stock['volume'].max()*4.0
        f = Figure(axes=[ax_x, ax_y], marks=[ohlc, bar],
                   fig_margin={"top": 0, "bottom": 60,
                               "left": 60, "right": 60})
        return {self.OUTPUT_PORT_NAME: f}
