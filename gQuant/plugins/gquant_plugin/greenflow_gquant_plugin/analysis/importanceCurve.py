from greenflow.dataframe_flow import Node, PortsSpecSchema
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from greenflow.dataframe_flow.portsSpecSchema import ConfSchema
from greenflow.dataframe_flow.metaSpec import MetaDataSchema
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin
from ..node_hdf_cache import NodeHDFCacheMixin


class ImportanceCurveNode(TemplateNodeMixin, NodeHDFCacheMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.INPUT_PORT_NAME = 'in'
        self.OUTPUT_PORT_NAME = 'importance_curve'
        port_type = PortsSpecSchema.port_type
        port_inports = {
            self.INPUT_PORT_NAME: {
                port_type: ["xgboost.Booster", "builtins.dict"]
            }
        }
        port_outports = {
            self.OUTPUT_PORT_NAME: {
                port_type: ["matplotlib.figure.Figure"]
            }
        }
        cols_required = {}
        retension = {}
        meta_inports = {
            self.INPUT_PORT_NAME: cols_required
        }
        meta_outports = {
            self.OUTPUT_PORT_NAME: {
                MetaDataSchema.META_OP: MetaDataSchema.META_OP_RETENTION,
                MetaDataSchema.META_DATA: retension
            }
        }
        self.template_ports_setup(
            in_ports=port_inports,
            out_ports=port_outports
        )
        self.template_meta_setup(
            in_ports=meta_inports,
            out_ports=meta_outports
        )

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
