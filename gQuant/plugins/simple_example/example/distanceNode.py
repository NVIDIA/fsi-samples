import numpy as np
from greenflow.dataframe_flow import Node
from greenflow.dataframe_flow import PortsSpecSchema
from greenflow.dataframe_flow import ConfSchema
from greenflow.dataframe_flow.metaSpec import MetaDataSchema
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin


class DistanceNode(TemplateNodeMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.delayed_process = True
        port_type = PortsSpecSchema.port_type
        self.INPUT_PORT_NAME = "points_df_in"
        self.OUTPUT_PORT_NAME = "distance_df"
        self.ABS_OUTPUT_PORT_NAME = "distance_abs_df"
        self.port_inports = {
            self.INPUT_PORT_NAME: {
                port_type: ["pandas.DataFrame"]
            }
        }
        self.port_outports = {
            self.OUTPUT_PORT_NAME: {
                port_type: "${port:points_df_in}"
            },
            self.ABS_OUTPUT_PORT_NAME: {
                port_type: "${port:points_df_in}"
            },
        }
        req_cols = {
            'x': 'float64',
            'y': 'float64'
        }
        self.meta_inports = {
            self.INPUT_PORT_NAME: req_cols
        }
        self.meta_outports = {
            self.OUTPUT_PORT_NAME: {
                MetaDataSchema.META_OP: MetaDataSchema.META_OP_ADDITION,
                MetaDataSchema.META_REF_INPUT: self.INPUT_PORT_NAME,
                MetaDataSchema.META_DATA: {
                    'distance_cudf': 'float64',
                }
            },
            self.ABS_OUTPUT_PORT_NAME: {
                MetaDataSchema.META_OP: MetaDataSchema.META_OP_ADDITION,
                MetaDataSchema.META_REF_INPUT: self.INPUT_PORT_NAME,
                MetaDataSchema.META_DATA: {
                    'distance_abs_cudf': 'float64',
                }
            }
        }

    def conf_schema(self):
        return ConfSchema()

    def process(self, inputs):
        df = inputs[self.INPUT_PORT_NAME]
        output = {}
        if self.outport_connected(self.OUTPUT_PORT_NAME):
            copy_df = df.copy()
            copy_df['distance_cudf'] = np.sqrt((df['x'] ** 2 + df['y'] ** 2))
            output.update({self.OUTPUT_PORT_NAME: copy_df})
        if self.outport_connected(self.ABS_OUTPUT_PORT_NAME):
            copy_df = df.copy()
            copy_df['distance_abs_cudf'] = np.abs(df['x']) + np.abs(df['y'])
            output.update({self.ABS_OUTPUT_PORT_NAME: copy_df})
        return output
