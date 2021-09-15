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
from dask.dataframe import DataFrame as DaskDataFrame


class RSquaredNode(TemplateNodeMixin, Node):

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
                port_type: "builtins.float"
            },
        }
        meta_inports = {
            self.INPUT_PORT_NAME: {}
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
            "title": "Compute the R-squared score for regression problems",
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": """Two columns used to compute the
                    R-squared score""",
                    "minItems": 2,
                    "maxItems": 2
                }
            },
            "required": ["columns"]
        }

        input_meta = self.get_input_meta()
        if self.INPUT_PORT_NAME in input_meta:
            col_from_inport = input_meta[self.INPUT_PORT_NAME]
            enums = [col for col in col_from_inport.keys()]
            json['properties']['columns']['items']['enum'] = enums
            ui = {}
            return ConfSchema(json=json, ui=ui)
        else:
            ui = {
                "column": {"ui:widget": "text"}
            }
            return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        df = inputs[self.INPUT_PORT_NAME]
        # df = df.drop('datetime', axis=1)
        output = {}
        subdf = df[self.conf['columns']]
        if isinstance(subdf, DaskDataFrame):
            result = subdf.corr().compute().values[0, 1]**2
        else:
            result = subdf.corr().values[0, 1]**2
        output.update({self.OUTPUT_PORT_NAME: result.item()})
        return output
