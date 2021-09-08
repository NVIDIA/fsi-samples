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
import math
import cupy
import cudf


class NRPWeightNode(TemplateNodeMixin, Node):

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
            },
        }
        self.template_ports_setup(in_ports=port_inports,
                                  out_ports=port_outports)

    def conf_schema(self):
        json = {
            "title": "Compute the Sharpe Ratio",
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
            assets = int(math.sqrt(len(input_meta[self.INPUT_PORT_NAME]) - 3))
            for i in range(assets):
                json[i] = 'float64'
        json.update(required)
        meta_outports[self.OUTPUT_PORT_NAME] = json
        self.template_meta_setup(
            in_ports=meta_inports,
            out_ports=meta_outports
        )

    def process(self, inputs):
        df = inputs[self.INPUT_PORT_NAME]

        all_sample_ids = df['sample_id'].unique()
        total_samples = len(all_sample_ids)

        # df = df.drop('datetime', axis=1)
        input_meta = self.get_input_meta()
        if self.INPUT_PORT_NAME in input_meta:
            assets = int(math.sqrt(len(input_meta[self.INPUT_PORT_NAME]) - 3))
        output = {}
        data_ma = df[list(range(assets*assets))].values
        data_ma = data_ma.reshape(total_samples, -1, assets, assets)
        diagonzied = cupy.diagonal(data_ma, 0, 2, 3)
        diagonzied = cupy.sqrt(1.0 / diagonzied)  # inverse variance
        diagonzied = diagonzied / diagonzied.sum(axis=2, keepdims=True)
        diagonzied = diagonzied.reshape(-1, assets)
        weight_df = cudf.DataFrame(diagonzied)
        weight_df['month'] = df['month']
        weight_df['year'] = df['year']
        weight_df['sample_id'] = df['sample_id']
        output.update({self.OUTPUT_PORT_NAME: weight_df})
        return output
