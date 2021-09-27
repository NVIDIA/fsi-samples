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
from greenflow.dataframe_flow.metaSpec import MetaDataSchema
from greenflow.dataframe_flow import Node
from .kernels import compute_leverage
import cupy
import cudf


class LeverageNode(TemplateNodeMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.delayed_process = True
        self.infer_meta = False
        self.LEVERAGE_DF = 'lev_df'
        self.INPUT_PORT_NAME = "in"
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
            self.LEVERAGE_DF: {
                port_type: "${port:in}"
            },
        }

        sub_dict = {
            "date": "datetime64[ns]",
            'sample_id': 'int64',
            'year': 'int16',
            'month': 'int16',
            'portfolio': "float64",
        }

        meta_inports = {
            self.INPUT_PORT_NAME: sub_dict
        }
        meta_outports = {
            self.LEVERAGE_DF: {
                MetaDataSchema.META_OP: MetaDataSchema.META_OP_RETENTION,
                MetaDataSchema.META_DATA: sub_dict
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
            "title": "Compute the Leverage to match the target volatility",
            "type": "object",
            "properties": {
                "target_vol": {
                    'type': "number",
                    "title": "Target Volativity",
                    "description": """The target volatility to match""",
                    "default": 0.05
                },
                "long_window": {
                    'type': "integer",
                    "title": "Long window size",
                    "description": """the large number of days in the past to compute
                     volatility""",
                    "default": 59
                },
                "short_window": {
                    'type': "integer",
                    "title": "Short window size",
                    "description": """the small number of days in the past to compute
                     volatility""",
                    "default": 19
                }
            },
        }

        ui = {
        }
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        df = inputs[self.INPUT_PORT_NAME]
        all_sample_ids = df['sample_id'].unique()
        total_samples = len(all_sample_ids)
        lev, all_dates, window = compute_leverage(total_samples, df,
                                                  **self.conf)

        total_samples, num_months = lev.shape

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
        df_lev = cudf.DataFrame(
            {'leverage': lev.reshape(total_samples * num_months)})
        df_lev['year'] = cupy.concatenate(
            [years]*total_samples).astype(cupy.int16)
        df_lev['month'] = cupy.concatenate(
            [months]*total_samples).astype(cupy.int16)
        df_lev['sample_id'] = cupy.repeat(cupy.arange(
            total_samples) + all_sample_ids.min(), len(mid))

        date_df = df[['date', 'sample_id', 'year', 'month', 'portfolio']]
        expand_table = date_df.reset_index().merge(
            df_lev, on=['sample_id', 'year', 'month'],
            how='left').set_index('index')
        expand_table['portfolio'] = expand_table[
            'portfolio'] * expand_table['leverage']
        expand_table = expand_table.dropna()[[
            'date', 'sample_id', 'year', 'month', 'portfolio'
        ]]
        output.update({self.LEVERAGE_DF: expand_table})
        return output
