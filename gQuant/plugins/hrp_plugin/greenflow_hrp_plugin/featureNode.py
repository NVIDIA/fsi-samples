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

from greenflow.dataframe_flow import (ConfSchema, PortsSpecSchema)
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin
from greenflow.dataframe_flow import Node


class FeatureNode(TemplateNodeMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.delayed_process = True
        self.infer_meta = False
        self.SIGNAL_DF = 'signal_df'
        self.FEATURE_DF = 'feature_df'
        self.OUTPUT_PORT_NAME = 'out'
        port_type = PortsSpecSchema.port_type
        port_inports = {
            self.SIGNAL_DF: {
                port_type: [
                    "pandas.DataFrame", "cudf.DataFrame",
                    "dask_cudf.DataFrame", "dask.dataframe.DataFrame"
                ]
            },
            self.FEATURE_DF: {
                port_type: [
                    "pandas.DataFrame", "cudf.DataFrame",
                    "dask_cudf.DataFrame", "dask.dataframe.DataFrame"
                ],
                PortsSpecSchema.optional: True
            }
        }
        port_outports = {
            self.OUTPUT_PORT_NAME: {
                port_type: "${port:signal_df}"
            },
        }
        self.template_ports_setup(in_ports=port_inports,
                                  out_ports=port_outports)

    def conf_schema(self):
        json = {
            "title": "Calculate the std and mean across assets as features",
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "title": "Feature Name"
                }
            },
        }

        ui = {
        }
        return ConfSchema(json=json, ui=ui)

    def update(self):
        TemplateNodeMixin.update(self)
        meta_outports = self.template_meta_setup().outports
        meta_inports = self.template_meta_setup().inports
        required = {
            'year': 'int16',
            'month': 'int16',
            'sample_id': 'int64',
        }
        name = self.conf.get("name", "feature")

        input_meta = self.get_input_meta()
        if self.FEATURE_DF not in input_meta:
            col_from_inport = required.copy()
        else:
            col_from_inport = input_meta[self.FEATURE_DF].copy()

        meta_inports[self.SIGNAL_DF] = required
        meta_inports[self.FEATURE_DF] = required

        # additional ports
        cols = {
            name+"_mean": "float64",
            name+"_std": "float64"
        }
        col_from_inport.update(cols)
        meta_outports[self.OUTPUT_PORT_NAME] = col_from_inport
        self.template_meta_setup(
            in_ports=meta_inports,
            out_ports=meta_outports
        )

    def process(self, inputs):
        df = inputs[self.SIGNAL_DF]
        name = self.conf.get("name", "feature")
        if self.FEATURE_DF not in inputs:
            output_df = df[['year', 'month', 'sample_id']].copy()
        else:
            output_df = inputs[self.FEATURE_DF]

        # df = df.drop('datetime', axis=1)
        output = {}
        if self.outport_connected(self.OUTPUT_PORT_NAME):
            col = list(df.columns)
            col.remove('sample_id')
            col.remove('year')
            col.remove('month')
            mean_val = df[col].values.mean(axis=1)
            std_val = df[col].values.std(axis=1)
            output_df[name+'_mean'] = mean_val
            output_df[name+'_std'] = std_val
            output.update({self.OUTPUT_PORT_NAME: output_df})
        return output
