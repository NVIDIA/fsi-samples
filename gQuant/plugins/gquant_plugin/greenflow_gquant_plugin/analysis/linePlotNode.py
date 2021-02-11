from greenflow.dataframe_flow import Node
from bqplot import Axis, LinearScale, DateScale, Figure, Lines, PanZoom
from greenflow.dataframe_flow.portsSpecSchema import ConfSchema, MetaData
import cudf
import dask_cudf
from .._port_type_node import _PortTypesMixin


class LinePlotNode(Node, _PortTypesMixin):

    def init(self):
        self.INPUT_PORT_NAME = 'in'
        self.OUTPUT_PORT_NAME = 'lineplot'

    def conf_schema(self):
        color_strings = ['black', 'yellow', 'blue',
                         'red', 'green', 'orange',
                         'magenta', 'cyan']
        json = {
            "title": "Line Plot Node Configuration",
            "type": "object",
            "description": """Plot the columns as lines""",
            "properties": {
                "points":  {
                    "type": "number",
                    "description": "number of data points for the chart"
                },
                "title":  {
                    "type": "string",
                    "description": "the plot title"
                },
                "lines": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "title": "Line Information",
                        "properties": {
                            "column": {
                                "type": "string",
                            },
                            "label": {
                                "type": "string",
                            },
                            "color": {
                                "type": "string",
                                "enum": color_strings
                            }
                        }
                    }
                }
            },
            "required": ["points", "title", "lines"],
        }
        input_meta = self.get_input_meta()
        ui = {
        }
        if self.INPUT_PORT_NAME in input_meta:
            col_inport = input_meta[self.INPUT_PORT_NAME]
            enums = [col for col in col_inport.keys()]
            first_item = json['properties']['lines']['items']
            first_item['properties']['column']['enum'] = enums
            return ConfSchema(json=json, ui=ui)
        else:
            return ConfSchema(json=json, ui=ui)

    def ports_setup(self):
        return _PortTypesMixin.ports_setup_different_output_type(self,
                                                                 Figure)

    def meta_setup(self):
        cols_required = {"datetime": "date"}
        required = {
            self.INPUT_PORT_NAME: cols_required
        }
        metadata = MetaData(inports=required,
                            outports={self.OUTPUT_PORT_NAME: {}})
        return metadata

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

        input_df = inputs[self.INPUT_PORT_NAME]

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
            if (isinstance(input_df,
                           cudf.DataFrame) or isinstance(input_df,
                                                         dask_cudf.DataFrame)):
                line = Lines(x=input_df['datetime'][::stride].to_array(),
                             y=input_df[col_name][::stride].to_array(),
                             scales={'x': date_co, 'y': linear_co},
                             colors=[color],
                             labels=[label_name], display_legend=True)
            else:
                line = Lines(x=input_df['datetime'][::stride],
                             y=input_df[col_name][::stride],
                             scales={'x': date_co, 'y': linear_co},
                             colors=[color],
                             labels=[label_name], display_legend=True)

            lines.append(line)
        new_fig = Figure(marks=lines, axes=[yax, xax],
                         title=self.conf['title'], interaction=panzoom_main)
        return {self.OUTPUT_PORT_NAME: new_fig}
