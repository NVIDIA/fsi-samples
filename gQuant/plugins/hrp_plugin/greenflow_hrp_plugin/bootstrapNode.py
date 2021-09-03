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

import cudf
from greenflow.dataframe_flow import Node
from greenflow.dataframe_flow import PortsSpecSchema
from greenflow.dataframe_flow import ConfSchema
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin
from .kernels import run_bootstrap
import cupy
import dask
import dask_cudf
from collections import OrderedDict


class BootstrapNode(TemplateNodeMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.INPUT_PORT_NAME = 'in'
        self.OUTPUT_PORT_NAME = 'out'
        self.OUTPUT_DASK_PORT = 'dask_df'
        port_type = PortsSpecSchema.port_type
        port_inports = {
            self.INPUT_PORT_NAME: {
                port_type: [
                    "pandas.DataFrame", "cudf.DataFrame",
                ]
            },
        }
        port_outports = {
            self.OUTPUT_PORT_NAME: {
                port_type: "${port:in}"
            },
            self.OUTPUT_DASK_PORT: {
                port_type: ["dask_cudf.DataFrame", "dask.dataframe.DataFrame"]
            }
        }
        self.template_ports_setup(
            in_ports=port_inports,
            out_ports=port_outports
        )

    def update(self):
        TemplateNodeMixin.update(self)
        meta_outports = self.template_meta_setup().outports
        meta_inports = self.template_meta_setup().inports
        col_required = {
            "date": "date"
        }
        input_meta = self.get_input_meta()
        json = OrderedDict()
        if self.INPUT_PORT_NAME in input_meta:
            assets = len(input_meta[self.INPUT_PORT_NAME]) - 1
            for i in range(assets):
                json[i] = 'float64'
        json['date'] = "datetime64[ns]"
        json['sample_id'] = 'int64'
        json['year'] = 'int16'
        json['month'] = 'int16'
        meta_inports[self.INPUT_PORT_NAME] = col_required
        meta_outports[self.OUTPUT_DASK_PORT] = json
        meta_outports[self.OUTPUT_PORT_NAME] = json
        self.template_meta_setup(
            in_ports=meta_inports,
            out_ports=meta_outports
        )

    def conf_schema(self):
        json = {
            "title": "Generate bootstrap dataframe",
            "type": "object",
            "properties": {
                "samples":  {
                    "type": "integer",
                    "description": "Number of samples to bootstrap"
                },
                "partitions":  {
                    "type": "integer",
                    "description": "Number of partitions for Dask Dataframe"
                },
                "offset":  {
                    "type": "integer",
                    "description": "Sample id offset",
                    "default": 0
                },
            },
            "required": ["samples"],
        }

        ui = {
        }
        return ConfSchema(json=json, ui=ui)

    def _process(self, df, partition_id):
        number_samples = self.conf['samples']
        all_dates = df['date']
        cols = list(df.columns)
        cols.remove('date')
        price_matrix = df[cols].values
        result = run_bootstrap(price_matrix, number_samples=number_samples)
        # print('bootstrap')
        total_samples, assets, length = result.shape
        datetime_col = cudf.concat([all_dates] *
                                   total_samples).reset_index(drop=True)
        result = result.transpose([0, 2, 1]).reshape(-1, assets)
        df = cudf.DataFrame(result)
        df['date'] = datetime_col
        sample_id = cupy.repeat(cupy.arange(0, total_samples), length)
        df['sample_id'] = sample_id + partition_id * number_samples
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month - 1
        return df

    def process(self, inputs):
        df = inputs[self.INPUT_PORT_NAME]
        # df = df.drop('datetime', axis=1)
        output = {}
        if self.outport_connected(self.OUTPUT_PORT_NAME):
            offset = self.conf.get('offset', 0)
            out_df = self._process(df, offset)
            output.update({self.OUTPUT_PORT_NAME: out_df})
        if self.outport_connected(self.OUTPUT_DASK_PORT):
            partitions = self.conf['partitions']
            out_dfs = [
                dask.delayed(self._process)(df, i) for i in range(partitions)
            ]
            meta = self.meta_setup().outports[self.OUTPUT_DASK_PORT]
            meta['date'] = 'datetime64[ns]'
            dask_df = dask_cudf.from_delayed(
                out_dfs, meta=meta)
            output.update({self.OUTPUT_DASK_PORT: dask_df})
        return output
