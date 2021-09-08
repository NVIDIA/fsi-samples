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
from greenflow.dataframe_flow import Node
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin
from .kernels import get_weights
import cudf
import math


class HRPWeightNode(TemplateNodeMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.COV_IN = 'covariance_df'
        self.ORDER_IN = 'asset_order_df'
        self.OUTPUT_PORT_NAME = 'out'
        self.delayed_process = True
        self.infer_meta = False
        port_type = PortsSpecSchema.port_type
        port_inports = {
            self.COV_IN: {
                port_type: [
                    "pandas.DataFrame", "cudf.DataFrame",
                    "dask_cudf.DataFrame", "dask.dataframe.DataFrame"
                ]
            },
            self.ORDER_IN: {
                port_type: [
                    "pandas.DataFrame", "cudf.DataFrame",
                    "dask_cudf.DataFrame", "dask.dataframe.DataFrame"
                ]
            }
        }
        port_outports = {
            self.OUTPUT_PORT_NAME: {
                port_type: "${port:covariance_df}"
            },
        }
        self.template_ports_setup(in_ports=port_inports,
                                  out_ports=port_outports)

    def conf_schema(self):
        json = {
            "title": "Compute the HRP weights",
            "type": "object",
            "properties": {
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
            'month': 'int16',
            'year': 'int16',
            'sample_id': 'int64',
        }
        meta_inports[self.COV_IN] = required
        meta_inports[self.ORDER_IN] = required
        json = {}
        input_meta = self.get_input_meta()
        if self.COV_IN in input_meta:
            assets = int(math.sqrt(len(input_meta[self.COV_IN]) - 3))
            for i in range(assets):
                json[i] = 'float64'
        elif self.ORDER_IN in input_meta:
            assets = len(input_meta[self.ORDER_IN]) - 3
            for i in range(assets):
                json[i] = 'float64'
        json.update(required)
        meta_outports[self.OUTPUT_PORT_NAME] = json
        self.template_meta_setup(
            in_ports=meta_inports,
            out_ports=meta_outports
        )

    def process(self, inputs):
        input_meta = self.get_input_meta()
        df_cov = inputs[self.COV_IN]
        df_order = inputs[self.ORDER_IN]
        all_sample_ids = df_cov['sample_id'].unique()
        # print(all_sample_ids - df_order['sample_id'].unique())
        total_samples = len(all_sample_ids)
        input_meta = self.get_input_meta()
        if self.COV_IN in input_meta:
            assets = int(math.sqrt(len(input_meta[self.COV_IN]) - 3))
        elif self.ORDER_IN in input_meta:
            assets = len(input_meta[self.ORDER_IN]) - 3

        output = {}
        col = list(df_cov.columns)
        col.remove('sample_id')
        col.remove('year')
        col.remove('month')
        cov = df_cov[col].values
        cov = cov.reshape(
            total_samples, -1, assets, assets)
        _, num_months, _, _ = cov.shape

        col = list(df_order.columns)
        col.remove('sample_id')
        col.remove('year')
        col.remove('month')
        order = df_order[col].values
        order = order.reshape(
            total_samples, -1, assets)
        weights = get_weights(total_samples, cov,
                              order, num_months, assets)
        weights = weights.reshape(-1, assets)
        weight_df = cudf.DataFrame(weights)
        weight_df['month'] = df_order['month']
        weight_df['year'] = df_order['year']
        weight_df['sample_id'] = df_order['sample_id']
        output.update({self.OUTPUT_PORT_NAME: weight_df})
        return output
