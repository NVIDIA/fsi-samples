from greenflow.dataframe_flow import Node
from bqplot import Axis, LinearScale,  Figure, Lines, PanZoom
import dask_cudf
import cudf
from greenflow.dataframe_flow.portsSpecSchema import (ConfSchema, PortsSpecSchema,
                                                   MetaData)
from .._port_type_node import _PortTypesMixin
from sklearn import metrics


class RocCurveNode(Node, _PortTypesMixin):

    def init(self):
        self.INPUT_PORT_NAME = 'in'
        self.OUTPUT_PORT_NAME = 'roc_curve'
        self.OUTPUT_VALUE_NAME = 'value'

    def meta_setup(self):
        cols_required = {}
        icols = self.get_input_meta()
        if 'label' in self.conf:
            label = self.conf['label']
            labeltype = icols.get(self.INPUT_PORT_NAME, {}).get(label)
            cols_required[label] = labeltype
        if 'prediction' in self.conf:
            cols_required[self.conf['prediction']] = None
        required = {
            self.INPUT_PORT_NAME: cols_required
        }
        metadata = MetaData(inports=required,
                            outports={self.OUTPUT_PORT_NAME: {},
                                      self.OUTPUT_VALUE_NAME: {}})
        return metadata

    def ports_setup(self):
        ports = _PortTypesMixin.ports_setup_different_output_type(self, Figure)
        ports.outports[self.OUTPUT_VALUE_NAME] = {PortsSpecSchema.port_type:
                                                  float}
        return ports

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

        if self.outport_connected(self.OUTPUT_PORT_NAME):
            linear_x = LinearScale()
            linear_y = LinearScale()
            yax = Axis(label='True Positive Rate', scale=linear_x,
                       orientation='vertical')
            xax = Axis(label='False Positive Rate', scale=linear_y,
                       orientation='horizontal')
            panzoom_main = PanZoom(scales={'x': [linear_x]})
            curve_label = 'ROC (area = {:.2f})'.format(auc_value)
            line = Lines(x=fpr, y=tpr,
                         scales={'x': linear_x, 'y': linear_y},
                         colors=['blue'], labels=[curve_label],
                         display_legend=True)
            new_fig = Figure(marks=[line], axes=[yax, xax], title='ROC Curve',
                             interaction=panzoom_main)
            out.update({self.OUTPUT_PORT_NAME: new_fig})
        if self.outport_connected(self.OUTPUT_VALUE_NAME):
            out.update({self.OUTPUT_VALUE_NAME: float(auc_value)})
        return out
