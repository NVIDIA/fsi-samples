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
from greenflow.dataframe_flow.metaSpec import MetaDataSchema
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin
from greenflow.dataframe_flow import Node
import math
import datetime
import cudf
import cupy
from .kernels import get_drawdown_metric


class PerformanceMetricNode(TemplateNodeMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.delayed_process = True
        self.infer_meta = False
        self.INPUT_PORT_NAME = 'in'
        self.RET_DF = 'ret_df'
        self.SD_DF = 'sd_df'
        self.SHARPE_DF = 'sharpe_df'
        self.CALMAR_DF = 'calmar_df'
        self.MDD_DF = 'maxdd_df'
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
            self.RET_DF: {
                port_type: "${port:in}"
            },
            self.SD_DF: {
                port_type: "${port:in}"
            },
            self.SHARPE_DF: {
                port_type: "${port:in}"
            },
            self.CALMAR_DF: {
                port_type: "${port:in}"
            },
            self.MDD_DF: {
                port_type: "${port:in}"
            }
        }
        required = {
            "date": "datetime64[ns]",
            'sample_id': 'int64',
            'portfolio': 'float64'
        }
        output = {
            'sample_id': 'int64',
            'portfolio': 'float64',
        }
        meta_inports = {
            self.INPUT_PORT_NAME: required
        }
        meta_outports = {
            self.RET_DF: {
                MetaDataSchema.META_OP: MetaDataSchema.META_OP_RETENTION,
                MetaDataSchema.META_DATA: output
            },
            self.SD_DF: {
                MetaDataSchema.META_OP: MetaDataSchema.META_OP_RETENTION,
                MetaDataSchema.META_DATA: output
            },
            self.SHARPE_DF: {
                MetaDataSchema.META_OP: MetaDataSchema.META_OP_RETENTION,
                MetaDataSchema.META_DATA: output
            },
            self.CALMAR_DF: {
                MetaDataSchema.META_OP: MetaDataSchema.META_OP_RETENTION,
                MetaDataSchema.META_DATA: output
            },
            self.MDD_DF: {
                MetaDataSchema.META_OP: MetaDataSchema.META_OP_RETENTION,
                MetaDataSchema.META_DATA: output
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
            "title": "Compute the Sharpe Ratio",
            "type": "object",
            "properties": {
            },
        }

        ui = {
        }
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        df = inputs[self.INPUT_PORT_NAME]
        # df = df.drop('datetime', axis=1)
        output = {}
        df = df.sort_values(['date'])
        group_obj = df.groupby('sample_id')
        beg = datetime.datetime.utcfromtimestamp(
            group_obj.nth(0)['date'].values[0].item() // 1e9)
        end = datetime.datetime.utcfromtimestamp(
            group_obj.nth(-1)['date'].values[0].item() // 1e9)
        total_days = (end - beg).days
        total = cudf.exp(group_obj['portfolio'].sum())
        avg_return = cupy.power(total, (365/total_days)) - 1.0
        return_series = cudf.Series(avg_return)
        return_series.index = total.index
        mean_df = cudf.DataFrame({'portfolio': return_series})
        # mean_df = df.groupby(['sample_id']).agg({'portfolio': 'mean'})
        std_df = df.groupby(['sample_id']).agg(
            {'portfolio': 'std'}) * math.sqrt(252)

        if self.outport_connected(self.SHARPE_DF):
            # sort by dates
            out_df = (mean_df / std_df).reset_index()
            output.update({self.SHARPE_DF: out_df})
        if self.outport_connected(self.SD_DF):
            output.update({self.SD_DF: std_df.reset_index()})
        if self.outport_connected(self.RET_DF):
            output.update({self.RET_DF: mean_df.reset_index()})
        if (self.outport_connected(self.MDD_DF) or
                self.outport_connected(self.CALMAR_DF)):
            all_sample_ids = df['sample_id'].unique()
            total_samples = len(all_sample_ids)
            drawdown, all_dates = get_drawdown_metric(df, total_samples)
            drawdown_series = cudf.Series(
                cupy.abs(drawdown.reshape(total_samples)))
            drawdown_series.index = mean_df.index
            drawdown_df = cudf.DataFrame({'portfolio': drawdown_series})
            if self.outport_connected(self.MDD_DF):
                output.update({self.MDD_DF: drawdown_df.reset_index()})
            if self.outport_connected(self.CALMAR_DF):
                calmar_df = (mean_df / drawdown_df).reset_index()
                output.update({self.CALMAR_DF: calmar_df})
        return output
