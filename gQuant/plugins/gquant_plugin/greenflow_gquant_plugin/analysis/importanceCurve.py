from greenflow.dataframe_flow import Node
# from bqplot import Axis, LinearScale,  Figure, OrdinalScale, Bars
import matplotlib as mpl
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
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
        # x_ord = OrdinalScale()
        # y_sc = LinearScale()
        backend_ = mpl.get_backend()
        mpl.use("Agg")  # Prevent showing stuff
        f = plt.figure()

        data = model.get_score(importance_type=self.conf.get('type', 'gain'))
        x_values = []
        y_values = []
        for key in data.keys():
            x_values.append(key)
            y_values.append(data[key])

        width = 0.35  # the width of the bars
        x = np.arange(len(x_values))
        plt.bar(x - width/2, y_values, width, label='Feature Importance')
        plt.xticks(x, x_values, rotation='vertical')
        # ax = f.get_axes()[0]
        # ax.set_xticks(x)
        # ax.set_xticklabels(x_values)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.grid(True)
        f.set_figwidth(15)
        f.set_figheight(8)
        mpl.use(backend_)
        return {self.OUTPUT_PORT_NAME: f}
