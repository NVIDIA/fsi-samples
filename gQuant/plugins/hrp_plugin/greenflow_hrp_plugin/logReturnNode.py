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
from greenflow.dataframe_flow.metaSpec import MetaDataSchema


class LogReturnNode(TemplateNodeMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.delayed_process = True
        self.infer_meta = False
        self.INPUT_PORT_NAME = "in"
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
                port_type: "${port:in}"
            },
        }
        required = {
            "date": "datetime64[ns]",
            'sample_id': 'int64',
            'year': 'int16',
            'month': 'int16',
        }
        meta_inports = {
            self.INPUT_PORT_NAME: required
        }
        meta_outports = {
            self.OUTPUT_PORT_NAME: {
                MetaDataSchema.META_OP: MetaDataSchema.META_OP_ADDITION,
                MetaDataSchema.META_REF_INPUT: self.INPUT_PORT_NAME,
                MetaDataSchema.META_DATA: {}
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
            "title": "Compute the log return dataframe",
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
        col = list(df.columns)
        col.remove('date')
        col.remove('sample_id')
        col.remove('year')
        col.remove('month')

        logprice = df[col].log()
        log_return = logprice - logprice.shift(1)
        log_return['date'] = df['date']
        log_return['sample_id'] = df['sample_id']
        log_return['year'] = df['year']
        log_return['month'] = df['month']
        log_return['corrupted'] = df['sample_id'] - \
            df['sample_id'].shift(1)
        log_return = log_return.dropna()
        corrupted = log_return['corrupted'] == 1
        # print('corruped rows', corrupted.sum())
        log_return[corrupted] = None
        log_return = log_return.dropna()
        log_return = log_return.drop('corrupted', axis=1)

        output.update({self.OUTPUT_PORT_NAME: log_return})
        return output
