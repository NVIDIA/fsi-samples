from gquant.dataframe_flow import Node
from bqplot import Axis, LinearScale, DateScale, Figure, OHLC, Bars, Tooltip
import cupy as cp


class BarPlotNode(Node):

    def columns_setup(self):
        self.required = {"datetime": "date",
                         "open": "float64",
                         "close": "float64",
                         "high": "float64",
                         "low": "float64",
                         "volume": "float64"}
        self.retention = {}

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
        stock = inputs[0]
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
        ohlc = OHLC(x=stock['datetime'][::stride].to_array(),
                    y=cp.asnumpy(stock[['open', 'high', 'low', 'close']]
                                 .as_gpu_matrix()[::stride, :]),
                    marker='candle', scales={'x': dt_scale, 'y': sc},
                    format='ohlc', stroke='blue',
                    display_legend=True, labels=[label])
        bar = Bars(x=stock['datetime'][::stride].to_array(),
                   y=stock['volume'][::stride].to_array(),
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
        return f


if __name__ == "__main__":
    from gquant.dataloader.csvStockLoader import CsvStockLoader
    from gquant.transform.averageNode import AverageNode
    from gquant.analysis.outCsvNode import OutCsvNode

    loader = CsvStockLoader("id0", {}, True, False)
    df = loader([])
    vf = AverageNode("id1", {"column": "volume"})
    df2 = vf([df])
    o = OutCsvNode("id3", {"path": "o.csv"})
    o([df2])
