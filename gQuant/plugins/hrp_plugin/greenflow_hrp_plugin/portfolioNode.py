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
import cudf
import cupy


class PortfolioNode(TemplateNodeMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.delayed_process = True
        self.infer_meta = False
        self.RETURN_IN = 'return_df'
        self.WEIGHT_IN = 'weight_df'
        self.TRANS_IN = 'transaction_df'
        self.OUTPUT_PORT_NAME = 'out'
        port_type = PortsSpecSchema.port_type
        port_inports = {
            self.RETURN_IN: {
                port_type: [
                    "pandas.DataFrame", "cudf.DataFrame",
                    "dask_cudf.DataFrame", "dask.dataframe.DataFrame"
                ]
            },
            self.WEIGHT_IN: {
                port_type: [
                    "pandas.DataFrame", "cudf.DataFrame",
                    "dask_cudf.DataFrame", "dask.dataframe.DataFrame"
                ]
            },
            self.TRANS_IN: {
                port_type: [
                    "pandas.DataFrame", "cudf.DataFrame",
                    "dask_cudf.DataFrame", "dask.dataframe.DataFrame"
                ]
            }
        }
        port_outports = {
            self.OUTPUT_PORT_NAME: {
                port_type: "${port:return_df}"
            },
        }
        self.template_ports_setup(in_ports=port_inports,
                                  out_ports=port_outports)

    def conf_schema(self):
        json = {
            "title": "Construct the portfolio",
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
        return_required = {
            "date": "datetime64[ns]",
            'sample_id': 'int64',
            'year': 'int16',
            'month': 'int16',
        }
        weight_required = {
            'sample_id': 'int64',
            'year': 'int16',
            'month': 'int16',
        }
        tran_required = {
            'sample_id': 'int64',
            'year': 'int16',
            'month': 'int16',
        }

        addition = {
            'portfolio':  'float64'
        }

        input_meta = self.get_input_meta()
        if self.RETURN_IN not in input_meta:
            col_from_inport = return_required.copy()
        else:
            col_from_inport = input_meta[self.RETURN_IN].copy()
        meta_inports[self.RETURN_IN] = return_required
        meta_inports[self.WEIGHT_IN] = weight_required
        meta_inports[self.TRANS_IN] = tran_required
        col_from_inport.update(addition)
        # additional ports
        meta_outports[self.OUTPUT_PORT_NAME] = col_from_inport
        self.template_meta_setup(
            in_ports=meta_inports,
            out_ports=meta_outports
        )

    def process(self, inputs):
        input_meta = self.get_input_meta()
        if self.RETURN_IN in input_meta:
            assets = len(input_meta[self.RETURN_IN]) - 4
        elif self.WEIGHT_IN in input_meta:
            assets = len(input_meta[self.WEIGHT_IN]) - 3
        elif self.TRANS_IN in input_meta:
            assets = len(input_meta[self.TRANS_IN]) - 3

        return_df = inputs[self.RETURN_IN]
        weight_df = inputs[self.WEIGHT_IN]
        date_df = return_df[['date', 'sample_id', 'year', 'month']]

        expand_table = date_df.reset_index().merge(
            weight_df, on=['sample_id', 'year', 'month'],
            how='left').set_index('index')

        price_table = return_df[list(range(assets))]
        weight_table = expand_table[list(range(assets))]

        if self.TRANS_IN in input_meta:
            tran_df = inputs[self.TRANS_IN]
            tran_expand_table = date_df.reset_index().merge(
                tran_df, on=['sample_id', 'year',
                             'month'], how='left').set_index('index')
            tran_expand_table = tran_expand_table.sort_index().dropna()
            months = (tran_expand_table['year'] * 12 +
                      tran_expand_table['month']).values
            months = ((months[1:] - months[:-1]) != 0).astype(cupy.float64)
            months = cupy.pad(months, ((1, 0)), mode='constant')
            months[0] = 1.0
            tran_table = tran_expand_table[list(range(assets))].values
            tran_table = tran_table * months[:, None]
            tran_table = cudf.DataFrame(tran_table)
            tran_table.index = tran_expand_table.index

            apply_table = (price_table * weight_table).sort_index().dropna()
            # hack to fix the column names
            apply_table.columns = list(range(assets))
            apply_weight = (apply_table - tran_table).sum(axis=1)
        else:
            apply_weight = (price_table * weight_table).sum(axis=1)

        return_df['portfolio'] = apply_weight.astype('float64')
        return_df = return_df.dropna()
        output = {}
        output.update({self.OUTPUT_PORT_NAME: return_df})
        return output
