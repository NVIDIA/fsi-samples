from gquant.dataframe_flow import Node
from bqplot import Axis, LinearScale, DateScale, Figure, Lines, PanZoom


class LinePlotNode(Node):

    def columns_setup(self):
        self.required = {"datetime": "date"}
        self.retentation = {}

    def process(self, inputs):
        """
        Plot the lines from the input dataframe. The plotted lines are the
        columns in the input dataframe which are specified in the `lines` of
        node's `conf`
        The plot title is defined in the `title` of the node's `conf`

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        Figure
        """

        input_df = inputs[0]

        num_points = self.conf['points']
        stride = max(len(input_df) // num_points, 1)
        date_co = DateScale()
        linear_co = LinearScale()
        yax = Axis(label='', scale=linear_co, orientation='vertical')
        xax = Axis(label='Time', scale=date_co, orientation='horizontal')
        panzoom_main = PanZoom(scales={'x': [date_co]})
        lines = []
        for line in self.conf['lines']:
            col_name = line['column']
            label_name = line['label']
            color = line['color']
            line = Lines(x=input_df['datetime'][::stride].to_array(),
                         y=input_df[col_name][::stride].to_array(),
                         scales={'x': date_co, 'y': linear_co}, colors=[color],
                         labels=[label_name], display_legend=True)
            lines.append(line)
        new_fig = Figure(marks=lines, axes=[yax, xax],
                         title=self.conf['title'], interaction=panzoom_main)
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
