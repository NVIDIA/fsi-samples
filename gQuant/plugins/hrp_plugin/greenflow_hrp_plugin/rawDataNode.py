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

from greenflow.dataframe_flow import Node
from greenflow.dataframe_flow import PortsSpecSchema
from greenflow.dataframe_flow import ConfSchema
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin
from collections import OrderedDict


class RawDataNode(TemplateNodeMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.INPUT_PORT_NAME = 'in'
        self.OUTPUT_PORT_NAME = 'out'
        port_type = PortsSpecSchema.port_type
        port_inports = {
            self.INPUT_PORT_NAME: {
                port_type: [
                    "pandas.DataFrame", "cudf.DataFrame"
                ]
            },
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
            "title": "Pass along the raw dataframe dataframe",
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
        meta_outports[self.INPUT_PORT_NAME] = required
        meta_outports[self.OUTPUT_PORT_NAME] = json
        self.template_meta_setup(
            in_ports=meta_inports,
            out_ports=meta_outports
        )

    def _process(self, df, partition_id):
        all_dates = df['date']
        cols = list(df.columns)
        cols.remove('date')
        df = df[cols]
        df.columns = list(range(len(df.columns)))
        df['date'] = all_dates
        df['sample_id'] = partition_id
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month - 1
        return df

    def process(self, inputs):
        df = inputs[self.INPUT_PORT_NAME]
        # df = df.drop('datetime', axis=1)
        output = {}
        offset = self.conf.get('offset', 0)
        out_df = self._process(df, offset)
        output.update({self.OUTPUT_PORT_NAME: out_df})
        return output
