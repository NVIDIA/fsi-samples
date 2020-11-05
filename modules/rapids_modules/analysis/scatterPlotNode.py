from gquant.dataframe_flow import Node
from bqplot import (Axis, LinearScale,  Figure,
                    DateScale, ColorScale, ColorAxis, Scatter)
import dask_cudf
import cudf
from gquant.dataframe_flow.portsSpecSchema import ConfSchema
from gquant.dataframe_flow._port_type_node import _PortTypesMixin

__all__ = ["ScatterPlotNode"]

scaleMap = {
    "ColorScale": ColorScale,
    "LinearScale": LinearScale,
    "DateScale": DateScale
}


class ScatterPlotNode(Node, _PortTypesMixin):

    def init(self):
        self.INPUT_PORT_NAME = 'in'
        self.OUTPUT_PORT_NAME = 'scatter_plot'

    def columns_setup(self):
        cols_required = {}
        if 'col_x' in self.conf:
            cols_required[self.conf['col_x']] = None
        if 'col_y' in self.conf:
            cols_required[self.conf['col_y']] = None
        if 'col_color' in self.conf:
            cols_required[self.conf['col_color']] = None
        self.required = {
            self.INPUT_PORT_NAME: cols_required
        }
        return {self.OUTPUT_PORT_NAME: {}}

    def ports_setup(self):
        return _PortTypesMixin.ports_setup_different_output_type(self,
                                                                 Figure)

    def conf_schema(self):
        json = {
            "title": "Scatter Plot Configuration",
            "type": "object",
            "description": """Make a Scatter Plot.
            """,
            "properties": {
                "points":  {
                    "type": "number",
                    "description": "number of data points for the chart"
                },
                "title":  {
                    "type": "string",
                    "description": "the plot title"
                },
                "col_x":  {
                    "type": "string",
                    "description": "column used for X-axis"
                },
                "col_x_scale":  {
                    "type": "string",
                    "description": "X-axis scale",
                    "enum": ["DateScale", "LinearScale"],
                    "default": "LinearScale"
                },
                "col_y":  {
                    "type": "string",
                    "description": "column used for Y-axis"
                },
                "col_y_scale":  {
                    "type": "string",
                    "description": "Y-axis scale",
                    "enum": ["DateScale", "LinearScale"],
                    "default": "LinearScale"
                },
                "col_color":  {
                    "type": "string",
                    "description": "column used for color"
                }
            },
            "required": ["points","title","col_x", "col_y"],
        }
        ui = {
        }
        input_columns = self.get_input_columns()
        if self.INPUT_PORT_NAME in input_columns:
            col_from_inport = input_columns[self.INPUT_PORT_NAME]
            enums = [col for col in col_from_inport.keys()]
            json['properties']['col_x']['enum'] = enums
            json['properties']['col_y']['enum'] = enums
            json['properties']['col_color']['enum'] = enums
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        Plot the Scatter plot

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
        num_points = self.conf['points']
        stride = max(len(input_df) // num_points, 1)

        sc_x = scaleMap[self.conf.get('col_x_scale', 'LinearScale')]()
        sc_y = scaleMap[self.conf.get('col_y_scale', 'LinearScale')]()

        x_col = self.conf['col_x']
        y_col = self.conf['col_y']
        ax_y = Axis(label=y_col, scale=sc_y, 
                    orientation='vertical', side='left')

        ax_x = Axis(label=x_col, scale=sc_x, num_ticks=10, label_location='end')
        m_chart = dict(top=50, bottom=70, left=50, right=100)
        if 'col_color' in self.conf:
            color_col = self.conf['col_color']
            sc_c1 = ColorScale()
            ax_c = ColorAxis(scale=sc_c1, tick_format='0.2%', label=color_col,
                             orientation='vertical', side='right')
            if (isinstance(input_df,
                           cudf.DataFrame) or isinstance(input_df,
                                                         dask_cudf.DataFrame)):
                scatter = Scatter(x=input_df[x_col][::stride].to_array(),
                                  y=input_df[y_col][::stride].to_array(),
                                  color=input_df[color_col][::stride].to_array(),
                                  scales={'x': sc_x, 'y': sc_y, 'color': sc_c1},
                                  stroke='black')
            else:
                scatter = Scatter(x=input_df[x_col][::stride],
                                  y=input_df[y_col][::stride],
                                  color=input_df[color_col][::stride],
                                  scales={'x': sc_x, 'y': sc_y, 'color': sc_c1},
                                  stroke='black')
            fig = Figure(axes=[ax_x, ax_c, ax_y], marks=[scatter],
                         fig_margin=m_chart,
                         title=self.conf['title'])

        else:
            if (isinstance(input_df,
                           cudf.DataFrame) or isinstance(input_df,
                                                         dask_cudf.DataFrame)):
                scatter = Scatter(x=input_df[x_col][::stride].to_array(),
                                  y=input_df[y_col][::stride].to_array(),
                                  scales={'x': sc_x, 'y': sc_y},
                                  stroke='black')
            else:
                scatter = Scatter(x=input_df[x_col][::stride],
                                  y=input_df[y_col][::stride],
                                  scales={'x': sc_x, 'y': sc_y},
                                  stroke='black')
            fig = Figure(axes=[ax_x, ax_y], marks=[scatter],
                         fig_margin=m_chart,
                         title=self.conf['title'])
        return {self.OUTPUT_PORT_NAME: fig}
