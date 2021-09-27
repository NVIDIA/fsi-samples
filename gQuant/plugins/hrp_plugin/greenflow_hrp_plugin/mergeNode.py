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
from greenflow.dataframe_flow.portsSpecSchema import ConfSchema
from greenflow.dataframe_flow.portsSpecSchema import PortsSpecSchema
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin
import cudf


class MergeNode(TemplateNodeMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.INPUT_PORT_LEFT_NAME = 'left'
        self.INPUT_PORT_RIGHT_NAME = 'right'
        self.OUTPUT_PORT_NAME = 'merged'
        self.delayed_process = True
        self.infer_meta = False
        port_type = PortsSpecSchema.port_type
        port_inports = {
            self.INPUT_PORT_LEFT_NAME: {
                port_type: [
                    "pandas.DataFrame", "cudf.DataFrame",
                    "dask_cudf.DataFrame", "dask.dataframe.DataFrame"
                ]
            },
            self.INPUT_PORT_RIGHT_NAME: {
                port_type: [
                    "pandas.DataFrame", "cudf.DataFrame",
                    "dask_cudf.DataFrame", "dask.dataframe.DataFrame"
                ]
            }
        }
        port_outports = {
            self.OUTPUT_PORT_NAME: {
                port_type: "${port:left}"
            },
        }
        self.template_ports_setup(in_ports=port_inports,
                                  out_ports=port_outports)

    def update(self):
        TemplateNodeMixin.update(self)
        meta_outports = self.template_meta_setup().outports
        meta_inports = self.template_meta_setup().inports
        cols_required = {}
        input_meta = self.get_input_meta()
        if (self.INPUT_PORT_LEFT_NAME in input_meta
                and self.INPUT_PORT_RIGHT_NAME in input_meta):
            col_from_left_inport = input_meta[self.INPUT_PORT_LEFT_NAME]
            col_from_right_inport = input_meta[self.INPUT_PORT_RIGHT_NAME]
            col_from_left_inport.update(col_from_right_inport)
            meta_outports[self.OUTPUT_PORT_NAME] = col_from_left_inport
        elif self.INPUT_PORT_LEFT_NAME in input_meta:
            col_from_left_inport = input_meta[self.INPUT_PORT_LEFT_NAME]
            meta_outports[self.OUTPUT_PORT_NAME] = col_from_left_inport
        elif self.INPUT_PORT_RIGHT_NAME in input_meta:
            col_from_right_inport = input_meta[self.INPUT_PORT_RIGHT_NAME]
            meta_outports[self.OUTPUT_PORT_NAME] = col_from_right_inport
        else:
            meta_outports[self.OUTPUT_PORT_NAME] = {}
        meta_inports[self.INPUT_PORT_RIGHT_NAME] = cols_required
        meta_inports[self.INPUT_PORT_LEFT_NAME] = cols_required
        self.template_meta_setup(
            in_ports=meta_inports,
            out_ports=meta_outports
        )

    def conf_schema(self):
        json = {
            "title": "DataFrame Merge configure",
            "type": "object",
            "description": """Merge two dataframes""",
            "properties": {
                "column":  {
                    "type": "string",
                    "description": "column name on which to do the merge"
                }
            },
            "required": ["column"],
        }
        input_meta = self.get_input_meta()
        if (self.INPUT_PORT_LEFT_NAME in input_meta
                and self.INPUT_PORT_RIGHT_NAME in input_meta):
            col_left_inport = input_meta[self.INPUT_PORT_LEFT_NAME]
            col_right_inport = input_meta[self.INPUT_PORT_RIGHT_NAME]
            enums1 = set([col for col in col_left_inport.keys()])
            enums2 = set([col for col in col_right_inport.keys()])
            json['properties']['column']['enum'] = list(
                enums1.intersection(enums2))
            ui = {}
            return ConfSchema(json=json, ui=ui)
        else:
            ui = {
                "column": {"ui:widget": "text"}
            }
            return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        left merge the two dataframes in the inputs. the `on column` is defined
        in the `column` of the node's conf

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        df1 = inputs[self.INPUT_PORT_LEFT_NAME]
        df2 = inputs[self.INPUT_PORT_RIGHT_NAME]
        return {self.OUTPUT_PORT_NAME: cudf.merge(df1, df2,
                                                  on=self.conf['column'],
                                                  how='inner')}
