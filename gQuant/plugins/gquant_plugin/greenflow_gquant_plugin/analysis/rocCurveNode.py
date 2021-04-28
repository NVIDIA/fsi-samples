from greenflow.dataframe_flow import Node
# from bqplot import Axis, LinearScale,  Figure, Lines, PanZoom
import dask_cudf
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import metrics
import cudf
from greenflow.dataframe_flow.portsSpecSchema import (ConfSchema,
                                                      PortsSpecSchema)
from greenflow.dataframe_flow.metaSpec import MetaDataSchema
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin
from ..node_hdf_cache import NodeHDFCacheMixin


class RocCurveNode(TemplateNodeMixin, NodeHDFCacheMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.INPUT_PORT_NAME = 'in'
        self.OUTPUT_PORT_NAME = 'roc_curve'
        self.OUTPUT_VALUE_NAME = 'value'
        port_type = PortsSpecSchema.port_type
        port_inports = {
            self.INPUT_PORT_NAME: {
                port_type: [
                    "pandas.DataFrame", "cudf.DataFrame",
                    "dask_cudf.DataFrame", "dask.dataframe.DataFrame"
                ]
            },
        }
        port_outports = {
            self.OUTPUT_PORT_NAME: {
                port_type: ["matplotlib.figure.Figure"]
            },
            self.OUTPUT_VALUE_NAME: {
                port_type: ["builtins.float"]
            }
        }
        cols_required = {}
        icols = self.get_input_meta()
        if 'label' in self.conf:
            label = self.conf['label']
            labeltype = icols.get(self.INPUT_PORT_NAME, {}).get(label)
            cols_required[label] = labeltype
        if 'prediction' in self.conf:
            cols_required[self.conf['prediction']] = None
        retension = {}
        meta_inports = {
            self.INPUT_PORT_NAME: cols_required
        }
        meta_outports = {
            self.OUTPUT_PORT_NAME: {
                MetaDataSchema.META_OP: MetaDataSchema.META_OP_RETENTION,
                MetaDataSchema.META_DATA: retension
            },
            self.OUTPUT_VALUE_NAME: {
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
            "title": "ROC Curve Configuration",
            "type": "object",
            "description": """Plot the ROC Curve for binary classification problem.
            """,
            "properties": {
                "label":  {
                    "type": "string",
                    "description": "Ground truth label column name"
                },
                "prediction":  {
                    "type": "string",
                    "description": "prediction probablity column"
                },

            },
            "required": ["label", "prediction"],
        }
        ui = {
        }
        input_meta = self.get_input_meta()
        if self.INPUT_PORT_NAME in input_meta:
            col_from_inport = input_meta[self.INPUT_PORT_NAME]
            enums = [col for col in col_from_inport.keys()]
            json['properties']['label']['enum'] = enums
            json['properties']['prediction']['enum'] = enums
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
        input_df = inputs[self.INPUT_PORT_NAME]
        if isinstance(input_df,  dask_cudf.DataFrame):
            input_df = input_df.compute()  # get the computed value

        label_col = input_df[self.conf['label']].values
        pred_col = input_df[self.conf['prediction']].values

        if isinstance(input_df, cudf.DataFrame):
            fpr, tpr, _ = metrics.roc_curve(label_col.get(),
                                            pred_col.get())
        else:
            fpr, tpr, _ = metrics.roc_curve(label_col,
                                            pred_col)
        auc_value = metrics.auc(fpr, tpr)
        out = {}
        backend_ = mpl.get_backend()
        mpl.use("Agg")  # Prevent showing stuff

        f = plt.figure()

        if self.outport_connected(self.OUTPUT_PORT_NAME):
            # linear_x = LinearScale()
            # linear_y = LinearScale()
            # yax = Axis(label='True Positive Rate', scale=linear_x,
            #            orientation='vertical')
            # xax = Axis(label='False Positive Rate', scale=linear_y,
            #            orientation='horizontal')
            # panzoom_main = PanZoom(scales={'x': [linear_x]})
            curve_label = 'ROC (area = {:.2f})'.format(auc_value)
            plt.plot(fpr, tpr, color='blue', label=curve_label)
            # line = Lines(x=fpr, y=tpr,
            #              scales={'x': linear_x, 'y': linear_y},
            #              colors=['blue'], labels=[curve_label],
            #              display_legend=True)
            # new_fig = Figure(marks=[line], axes=[yax, xax],
            #                  title='ROC Curve',
            #                  interaction=panzoom_main)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.grid(True)
            plt.title('ROC Curve')
            plt.legend()
            mpl.use(backend_)
            out.update({self.OUTPUT_PORT_NAME: f})
        if self.outport_connected(self.OUTPUT_VALUE_NAME):
            out.update({self.OUTPUT_VALUE_NAME: float(auc_value)})
        return out
