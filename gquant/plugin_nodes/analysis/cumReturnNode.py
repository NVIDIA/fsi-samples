from gquant.dataframe_flow import Node
from bqplot import Axis, LinearScale, DateScale, Figure, Lines, PanZoom
import dask_cudf


class CumReturnNode(Node):

    def columns_setup(self):
        self.required = {"datetime": "datetime64[ms]",
                         "strategy_returns": "float64"}

        self.retention = {}

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
        input_df = inputs[0]
        if isinstance(input_df,  dask_cudf.DataFrame):
            input_df = input_df.compute()  # get the computed value
        label = 'stock'
        if 'label' in self.conf:
            label = self.conf['label']
        num_points = self.conf['points']
        stride = max(len(input_df) // num_points, 1)
        date_co = DateScale()
        linear_co = LinearScale()
        yax = Axis(label='Cumulative return', scale=linear_co,
                   orientation='vertical')
        xax = Axis(label='Time', scale=date_co, orientation='horizontal')
        panzoom_main = PanZoom(scales={'x': [date_co]})
        line = Lines(x=input_df['datetime'][::stride],
                     y=(input_df['strategy_returns'].cumsum())[::stride],
                     scales={'x': date_co, 'y': linear_co},
                     colors=['blue'], labels=[label], display_legend=True)
        new_fig = Figure(marks=[line], axes=[yax, xax], title='P & L',
                         interaction=panzoom_main)
        return new_fig


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
