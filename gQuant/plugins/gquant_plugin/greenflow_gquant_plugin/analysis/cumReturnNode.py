from greenflow.dataframe_flow import Node
from bqplot import Axis, LinearScale, DateScale, Figure, Lines, PanZoom
import dask_cudf
import cudf
from greenflow.dataframe_flow.portsSpecSchema import ConfSchema, MetaData
from .._port_type_node import _PortTypesMixin


class CumReturnNode(Node, _PortTypesMixin):

    def init(self):
        self.INPUT_PORT_NAME = 'in'
        self.OUTPUT_PORT_NAME = 'cum_return'

    def meta_setup(self):
        cols_required = {"datetime": "date",
                         "strategy_returns": "float64"}
        required = {
            self.INPUT_PORT_NAME: cols_required
        }
        metadata = MetaData(inports=required,
                            outports={self.OUTPUT_PORT_NAME: {}})
        return metadata

    def ports_setup(self):
        return _PortTypesMixin.ports_setup_different_output_type(self,
                                                                 Figure)

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
        if (isinstance(input_df,
                       cudf.DataFrame) or isinstance(input_df,
                                                     dask_cudf.DataFrame)):
            line = Lines(x=input_df['datetime'][::stride].to_array(),
                         y=(input_df[
                             'strategy_returns'].cumsum())[
                                 ::stride].to_array(),
                         scales={'x': date_co, 'y': linear_co},
                         colors=['blue'], labels=[label], display_legend=True)
        else:
            line = Lines(x=input_df['datetime'][::stride],
                         y=(input_df[
                             'strategy_returns'].cumsum())[::stride],
                         scales={'x': date_co, 'y': linear_co},
                         colors=['blue'], labels=[label], display_legend=True)
        new_fig = Figure(marks=[line], axes=[yax, xax], title='P & L',
                         interaction=panzoom_main)
        return {self.OUTPUT_PORT_NAME: new_fig}
