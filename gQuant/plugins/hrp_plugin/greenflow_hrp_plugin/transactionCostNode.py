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
import cupy
import cudf


class TransactionCostNode(TemplateNodeMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.delayed_process = True
        self.infer_meta = False
        self.INPUT_PORT_NAME = 'logreturn_df'
        self.OUTPUT_PORT_NAME = 'out'
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
                port_type: "${port:logreturn_df}"
            },
        }
        self.template_ports_setup(in_ports=port_inports,
                                  out_ports=port_outports)

    def conf_schema(self):
        json = {
            "title": "Compute the Transaction Cost",
            "type": "object",
            "properties": {
                "cost": {
                    'type': "number",
                    "title": "transaction cost",
                    "default": 2e-4
                },
            },
        }

        ui = {
        }
        return ConfSchema(json=json, ui=ui)

    def update(self):
        TemplateNodeMixin.update(self)
        meta_outports = self.template_meta_setup().outports
        meta_inports = self.template_meta_setup().inports
        sub_dict = {
            'year': 'int16',
            'month': 'int16',
            'sample_id': 'int64',
        }
        required = {
        }
        required.update(sub_dict)
        meta_inports[self.INPUT_PORT_NAME] = required
        json_drawdown = {}
        input_meta = self.get_input_meta()
        if self.INPUT_PORT_NAME in input_meta:
            assets = len(input_meta[self.INPUT_PORT_NAME]) - 3
            for i in range(assets):
                json_drawdown[i] = 'float64'
        json_drawdown.update(sub_dict)
        meta_outports[self.OUTPUT_PORT_NAME] = json_drawdown
        self.template_meta_setup(
            in_ports=meta_inports,
            out_ports=meta_outports
        )

    def process(self, inputs):
        df = inputs[self.INPUT_PORT_NAME]
        input_meta = self.get_input_meta()
        assets = len(input_meta[self.INPUT_PORT_NAME]) - 3
        all_sample_ids = df['sample_id'].unique()
        total_samples = len(all_sample_ids)
        cost = self.conf.get('cost', 2e-4)
        data = df[list(range(assets))].values
        r = data.reshape(total_samples, -1, assets)
        tcost = cupy.abs(r[:, 1:, :] - r[:, :-1, :])
        tcost = cupy.pad(tcost, ((0, 0), (1, 0), (0, 0)), mode='constant')
        tcost = tcost * cost
        tcost = tcost.reshape(-1, assets)
        cost_df = cudf.DataFrame(tcost)
        cost_df.index = df.index
        cost_df['year'] = df['year']
        cost_df['month'] = df['month']
        cost_df['sample_id'] = df['sample_id']
        output = {}
        output.update({self.OUTPUT_PORT_NAME: cost_df})
        return output
