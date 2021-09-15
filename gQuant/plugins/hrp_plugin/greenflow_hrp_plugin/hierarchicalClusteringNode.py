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

from greenflow.dataframe_flow import ConfSchema, PortsSpecSchema
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin
from greenflow.dataframe_flow import Node
from .kernels import get_orders
import math
import cudf


class HierarchicalClusteringNode(TemplateNodeMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.delayed_process = True
        self.infer_meta = False
        self.INPUT_PORT_NAME = 'in'
        self.OUTPUT_PORT_NAME = 'out'
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
            },
        }
        self.template_ports_setup(in_ports=port_inports,
                                  out_ports=port_outports)

    def conf_schema(self):
        json = {
            "title": "Hierachical Clustering Node",
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
        meta_inports[self.INPUT_PORT_NAME] = required
        json = {}
        input_meta = self.get_input_meta()
        if self.INPUT_PORT_NAME in input_meta:
            num = len(input_meta[self.INPUT_PORT_NAME]) - 3
            assets = (1 + int(math.sqrt(1 + 8 * num))) // 2
            for i in range(assets):
                json[i] = 'int64'
        json.update(required)
        meta_outports[self.OUTPUT_PORT_NAME] = json
        self.template_meta_setup(
            in_ports=meta_inports,
            out_ports=meta_outports
        )

    def process(self, inputs):
        input_meta = self.get_input_meta()
        df = inputs[self.INPUT_PORT_NAME]
        all_sample_ids = df['sample_id'].unique()
        total_samples = len(all_sample_ids)
        if self.INPUT_PORT_NAME in input_meta:
            num = len(input_meta[self.INPUT_PORT_NAME]) - 3
            assets = (1 + int(math.sqrt(1 + 8 * num))) // 2
        df = inputs[self.INPUT_PORT_NAME]

        output = {}
        col = list(df.columns)
        col.remove('sample_id')
        col.remove('year')
        col.remove('month')
        distance = df[col].values
        distance = distance.reshape(
            total_samples, -1, assets*(assets-1)//2)
        _, num_months, _ = distance.shape
        orders = get_orders(total_samples, num_months, assets, distance)
        orders = orders.reshape(-1, assets)
        order_df = cudf.DataFrame(orders)
        order_df['month'] = df['month']
        order_df['year'] = df['year']
        order_df['sample_id'] = df['sample_id']
        output.update({self.OUTPUT_PORT_NAME: order_df})
        return output
