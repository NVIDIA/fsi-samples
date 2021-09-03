"""
 ////////////////////////////////////////////////////////////////////////////
 //
 // Copyright (C) NVIDIA Corporation.  All rights reserved.
 //
 // NVIDIA Sample Code
 //
 // Please refer to the NVIDIA end user license agreement (EULA) associated
 // with this source code for terms and conditions that govern your use of
 // this software. Any use, reproduction, disclosure, or distribution of
 // this software and related documentation outside the terms of the EULA
 // is strictly prohibited.
 //
 ////////////////////////////////////////////////////////////////////////////
"""

from greenflow.dataframe_flow import Node
from greenflow.dataframe_flow.portsSpecSchema import (ConfSchema,
                                                      PortsSpecSchema,
                                                      NodePorts)
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin
from greenflow.dataframe_flow.metaSpec import MetaDataSchema
import cudf
from xgboost import Booster
import pandas as pd
from matplotlib.figure import Figure
from dask.dataframe import DataFrame as DaskDataFrame
import shap


class ShapSummaryPlotPlotNode(TemplateNodeMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.SHAP_INPUT_PORT_NAME = 'shap_in'
        self.MODEL_INPUT_PORT_NAME = 'model_in'
        self.DATA_INPUT_PORT_NAME = 'data_in'
        self.OUTPUT_PORT_NAME = 'summary_plot'
        port_type = PortsSpecSchema.port_type
        port_inports = {
            self.SHAP_INPUT_PORT_NAME: {
                port_type: [
                    "pandas.DataFrame", "cudf.DataFrame",
                    "dask_cudf.DataFrame", "dask.dataframe.DataFrame"
                ]
            },
            self.MODEL_INPUT_PORT_NAME: {
                port_type: [
                    "xgboost.Booster", "builtins.dict",
                    "dask_cudf.DataFrame", "dask.dataframe.DataFrame"
                ]
            },
            self.DATA_INPUT_PORT_NAME: {
                port_type: [
                    "pandas.DataFrame", "cudf.DataFrame",
                    "dask_cudf.DataFrame", "dask.dataframe.DataFrame"
                ]
            }
        }
        port_outports = {
            self.OUTPUT_PORT_NAME: {
                port_type: "matplotlib.figure.Figure"
            },
        }
        meta_inports = {
            self.MODEL_INPUT_PORT_NAME: {},
            self.DATA_INPUT_PORT_NAME: {},
            self.SHAP_INPUT_PORT_NAME: {}
        }
        meta_outports = {
            self.OUTPUT_PORT_NAME: {
                MetaDataSchema.META_OP: MetaDataSchema.META_OP_RETENTION,
                MetaDataSchema.META_DATA: {}
            }
        }
        self.template_ports_setup(in_ports=port_inports,
                                  out_ports=port_outports)
        self.template_meta_setup(
            in_ports=meta_inports,
            out_ports=meta_outports
        )

    def conf_schema(self):
        json = {
            "title": "Shap Summary Plot Node",
            "type": "object",
            "description": """Plot the Shap summary""",
            "properties": {
                  "max_display": {
                      "type": "integer",
                      "description": """
                       How many top features to include in the plot
                       (default is 20, or 7 for interaction plots)
                      """,
                      "default": 20
                  },
                  "plot_type": {
                      "type": "string",
                      "description": """
                      "dot" (default for single output), "bar" (default for
                       multi-output), "violin",
                      """,
                      "enum": ["dot", "bar", "violin"]
                  }
            }
        }
        # input_meta = self.get_input_meta()
        ui = {
        }
        return ConfSchema(json=json, ui=ui)

    def ports_setup(self):
        types = [cudf.DataFrame,
                 DaskDataFrame,
                 pd.DataFrame]
        port_type = PortsSpecSchema.port_type
        input_ports = {
            self.SHAP_INPUT_PORT_NAME: {
                port_type: types
            },
            self.MODEL_INPUT_PORT_NAME: {
                port_type: [Booster, dict]
            },
            self.DATA_INPUT_PORT_NAME: {
                 port_type: types
            }
        }
        output_ports = {
            self.OUTPUT_PORT_NAME: {
                port_type: Figure
            }
        }
        input_connections = self.get_connected_inports()
        if (self.SHAP_INPUT_PORT_NAME in input_connections):
            determined_type = input_connections[self.SHAP_INPUT_PORT_NAME]
            input_ports[self.SHAP_INPUT_PORT_NAME] = {
                port_type: determined_type
            }
        if (self.DATA_INPUT_PORT_NAME in input_connections):
            determined_type = input_connections[self.DATA_INPUT_PORT_NAME]
            input_ports[self.DATA_INPUT_PORT_NAME] = {
                port_type: determined_type
            }
        if (self.MODEL_INPUT_PORT_NAME in input_connections):
            determined_type = input_connections[self.MODEL_INPUT_PORT_NAME]
            input_ports[self.MODEL_INPUT_PORT_NAME] = {
                port_type: determined_type
            }
        ports = NodePorts(inports=input_ports, outports=output_ports)
        return ports

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
        import matplotlib.pyplot as pl
        pl.figure()
        shap_values = inputs[self.SHAP_INPUT_PORT_NAME]
        df = inputs[self.DATA_INPUT_PORT_NAME]
        if isinstance(shap_values, DaskDataFrame):
            shap_values = shap_values.compute()
        if isinstance(df, DaskDataFrame):
            df = df.compute()
        if isinstance(shap_values, cudf.DataFrame):
            shap_values = shap_values.values.get()
        else:
            shap_values = shap_values.values
        if isinstance(df, cudf.DataFrame):
            df = df.to_pandas()
        input_meta = self.get_input_meta()
        required_cols = input_meta[
            self.MODEL_INPUT_PORT_NAME]['train']
        df = df[required_cols]
        self.conf['show'] = False
        # max_display = self.conf.get('max_display', 20)
        # plot_type = self.conf.get('plot_type', 'bar')
        shap.summary_plot(shap_values[:, :-1],
                          df, **self.conf)
        f = pl.gcf()
        return {self.OUTPUT_PORT_NAME: f}
