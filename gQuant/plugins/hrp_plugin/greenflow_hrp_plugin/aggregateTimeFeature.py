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

from greenflow.dataframe_flow import ConfSchema
from greenflow.dataframe_flow.portsSpecSchema import PortsSpecSchema
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin
from greenflow.dataframe_flow.metaSpec import MetaDataSchema
from greenflow.dataframe_flow import Node
import cudf


class AggregateTimeFeatureNode(TemplateNodeMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.INPUT_PORT_NAME = 'in'
        self.OUTPUT_PORT_NAME = 'out'
        self.delayed_process = True
        self.infer_meta = False
        port_type = PortsSpecSchema.port_type
        port_inports = {
            self.INPUT_PORT_NAME: {
                port_type: [
                    "pandas.DataFrame", "cudf.DataFrame",
                    "dask_cudf.DataFrame", "dask.dataframe.DataFrame"
                ]
            }
        }
        port_outports = {
            self.OUTPUT_PORT_NAME: {
                port_type: "${port:in}"
            }
        }
        cols_required = {
            'sample_id': 'int64',
            'year': 'int16',
            'month': 'int16',
        }

        meta_inports = {
            self.INPUT_PORT_NAME: cols_required
        }
        meta_outports = {
            self.OUTPUT_PORT_NAME: {
                MetaDataSchema.META_OP: MetaDataSchema.META_OP_RETENTION,
                MetaDataSchema.META_DATA: {}
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

    def update(self):
        TemplateNodeMixin.update(self)
        input_meta = self.get_input_meta()
        meta_outports = self.template_meta_setup().outports
        meta_inports = self.template_meta_setup().inports
        required = meta_inports[self.INPUT_PORT_NAME]
        if self.INPUT_PORT_NAME not in input_meta:
            col_from_inport = {
                'sample_id': 'int64'
            }
            col_ref = {}
        else:
            col_from_inport = {
                'sample_id': 'int64'
            }
            col_ref = input_meta[self.INPUT_PORT_NAME].copy()

        for key in col_ref.keys():
            if key in required:
                continue
            new_key = key+"_mean"
            col_from_inport[new_key] = col_ref[key]
        for key in col_ref.keys():
            if key in required:
                continue
            new_key = key+"_std"
            col_from_inport[new_key] = col_ref[key]
        meta_outports[self.OUTPUT_PORT_NAME] = col_from_inport
        self.template_meta_setup(
            in_ports=None,
            out_ports=meta_outports
        )

    def conf_schema(self):
        json = {
            "title": "Aggregate feature across time, get mean and std",
            "type": "object",
            "properties": {
            },
        }

        ui = {
        }
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        df = inputs[self.INPUT_PORT_NAME]
        output = {}

        col = list(df.columns)
        col.remove('year')
        col.remove('month')

        mdf = df[col].groupby('sample_id').mean()
        mdf.columns = [c+"_mean" for c in mdf.columns]

        sdf = df[col].groupby('sample_id').std()
        sdf.columns = [c+"_std" for c in sdf.columns]

        out = cudf.merge(mdf, sdf,
                         left_index=True,
                         right_index=True).reset_index()
        output.update({self.OUTPUT_PORT_NAME: out})
        return output
