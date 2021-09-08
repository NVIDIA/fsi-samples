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
from greenflow.dataframe_flow.metaSpec import MetaDataSchema


class DiffNode(TemplateNodeMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.delayed_process = True
        self.infer_meta = False
        self.OUTPUT_PORT_NAME = 'out'
        self.DIFF_A = 'diff_a'
        self.DIFF_B = 'diff_b'
        port_type = PortsSpecSchema.port_type
        port_inports = {
            self.DIFF_A: {
                port_type: [
                    "pandas.DataFrame", "cudf.DataFrame",
                    "dask_cudf.DataFrame", "dask.dataframe.DataFrame"
                ]
            },
            self.DIFF_B: {
                port_type: [
                    "pandas.DataFrame", "cudf.DataFrame",
                    "dask_cudf.DataFrame", "dask.dataframe.DataFrame"
                ]
            }
        }
        port_outports = {
            self.OUTPUT_PORT_NAME: {
                port_type: "${port:diff_a}"
            },
        }
        col_required = {
            'sample_id': 'int64',
            'portfolio': 'float64',
        }

        meta_inports = {
            self.DIFF_A: col_required,
            self.DIFF_B: col_required
        }
        output_meta = {
            'sample_id': 'int64',
            'portfolio': 'float64',
        }
        meta_outports = {
            self.OUTPUT_PORT_NAME: {
                MetaDataSchema.META_OP: MetaDataSchema.META_OP_RETENTION,
                MetaDataSchema.META_DATA: output_meta
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
            "title": "Calculate Sharpe diff",
            "type": "object",
            "properties": {
            },
        }

        ui = {
        }
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        df_a = inputs[self.DIFF_A].set_index('sample_id')
        df_b = inputs[self.DIFF_B].set_index('sample_id')

        # df = df.drop('datetime', axis=1)
        output = {}
        diff = df_a - df_b
        output.update({self.OUTPUT_PORT_NAME: diff.reset_index()})
        return output
