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
from .kernels import get_drawdown
import cupy
import cudf


class MaxDrawdownNode(TemplateNodeMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.delayed_process = True
        self.infer_meta = False
        self.INPUT_PORT_NAME = 'logreturn_df'
        self.OUTPUT_PORT_NAME = "out"
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
                port_type: "${port:logreturn_df}"
            },
        }
        self.template_ports_setup(in_ports=port_inports,
                                  out_ports=port_outports)

    def conf_schema(self):
        json = {
            "title": "Compute the Maximum Drawdown Matrix Dataframe",
            "type": "object",
            "properties": {
                "window": {
                    'type': "integer",
                    "title": "Window size",
                    "description": """the number of months used to compute the
                    distance and vairance"""
                },
                "negative": {
                    'type': "boolean",
                    "title": "Negative return",
                    "description": """Compute
                     max drawdown on negative return""",
                    "default": False
                }

            },
            "required": ["window"],
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
            "date": "datetime64[ns]",
        }
        required.update(sub_dict)
        meta_inports[self.INPUT_PORT_NAME] = required
        json_drawdown = {}
        input_meta = self.get_input_meta()
        if self.INPUT_PORT_NAME in input_meta:
            assets = len(input_meta[self.INPUT_PORT_NAME]) - 4
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
        all_sample_ids = df['sample_id'].unique()
        total_samples = len(all_sample_ids)
        window = self.conf['window']
        negative = self.conf.get("negative", False)
        drawdown, all_dates = get_drawdown(df, total_samples,
                                           negative=negative, window=window)

        total_samples, num_months, assets = drawdown.shape

        months_id = all_dates.dt.year*12 + (all_dates.dt.month-1)
        months_id = months_id - months_id.min()
        mid = (cupy.arange(months_id.max() + 1) +
               (all_dates.dt.month - 1)[0])[window:]
        minyear = all_dates.dt.year.min()
        if len(mid) == 0:
            mid = cupy.array([0])
        months = mid % 12
        years = mid // 12 + minyear

        output = {}
        df_drawdown = cudf.DataFrame(
            drawdown.reshape(total_samples*num_months, -1))
        df_drawdown['year'] = cupy.concatenate(
            [years]*total_samples).astype(cupy.int16)
        df_drawdown['month'] = cupy.concatenate(
            [months]*total_samples).astype(cupy.int16)
        df_drawdown['sample_id'] = cupy.repeat(cupy.arange(
            total_samples) + all_sample_ids.min(), len(mid))
        output.update({self.OUTPUT_PORT_NAME: df_drawdown})
        return output
