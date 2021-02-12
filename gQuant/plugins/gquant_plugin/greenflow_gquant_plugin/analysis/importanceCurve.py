from greenflow.dataframe_flow import Node
from bqplot import Axis, LinearScale,  Figure, OrdinalScale, Bars
from greenflow.dataframe_flow.portsSpecSchema import (ConfSchema, NodePorts,
                                                   MetaData,
                                                   PortsSpecSchema)
from xgboost import Booster


class ImportanceCurveNode(Node):

    def init(self):
        self.INPUT_PORT_NAME = 'in'
        self.OUTPUT_PORT_NAME = 'importance_curve'

    def meta_setup(self):
        cols_required = {}
        required = {
            self.INPUT_PORT_NAME: cols_required
        }
        metadata = MetaData(inports=required,
                            outports={self.OUTPUT_PORT_NAME: {}})
        return metadata

    def ports_setup(self):
        port_type = PortsSpecSchema.port_type
        output_ports = {
            self.OUTPUT_PORT_NAME: {
                port_type: Figure
            }
        }
        input_ports = {
            self.INPUT_PORT_NAME: {
                port_type: [Booster, dict]
            }
        }
        return NodePorts(inports=input_ports, outports=output_ports)

    def conf_schema(self):
        json = {
            "title": "Feature Importance Plot Configuration",
            "type": "object",
            "description": """Plot feature importance of each feature.
            """,
            "properties": {
                "type":  {
                    "type": "string",
                    "description": """
                        * 'weight': the number of times a feature is used to
                                    split the data across all trees.
                        * 'gain': the average gain across all splits the
                                    feature is used in.
                        * 'cover': the average coverage across all
                                   splits the feature is used in.
                        * 'total_gain': the total gain across all splits the
                                        feature is used in.
                        * 'total_cover': the total coverage across all splits
                                         the feature is used in.
                    """,
                    "enum": ["weight", "gain", "cover",
                             "total_gain", "total_cover"],
                    "default": "gain"
                },
            },
            "required": ["type"],
        }
        ui = {
        }
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        Plot the ROC curve

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        Figure

        """
        model = inputs[self.INPUT_PORT_NAME]
        if isinstance(model, dict):
            model = model['booster']
        x_ord = OrdinalScale()
        y_sc = LinearScale()
        data = model.get_score(importance_type=self.conf.get('type', 'gain'))
        x_values = []
        y_values = []
        for key in data.keys():
            x_values.append(key)
            y_values.append(data[key])

        bar = Bars(x=x_values, y=y_values, scales={'x': x_ord, 'y': y_sc})
        ax_x = Axis(scale=x_ord)
        ax_y = Axis(scale=y_sc, tick_format='0.2f', orientation='vertical')
        newFig = Figure(marks=[bar], axes=[ax_x, ax_y], padding_x=0.025,
                        padding_y=0.025)
        return {self.OUTPUT_PORT_NAME: newFig}
