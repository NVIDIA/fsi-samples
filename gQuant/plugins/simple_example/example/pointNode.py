import numpy as np
import pandas as pd
from greenflow.dataframe_flow import Node
from greenflow.dataframe_flow import PortsSpecSchema
from greenflow.dataframe_flow import ConfSchema
from greenflow.dataframe_flow.metaSpec import MetaDataSchema
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin


class PointNode(TemplateNodeMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.OUTPUT_PORT_NAME = 'points_df_out'
        self.port_inports = {}
        self.port_outports = {
            self.OUTPUT_PORT_NAME: {
                PortsSpecSchema.port_type: ["pandas.DataFrame"]
            }
        }
        self.meta_inports = {}
        self.meta_outports = {
            self.OUTPUT_PORT_NAME: {
                MetaDataSchema.META_OP: MetaDataSchema.META_OP_RETENTION,
                MetaDataSchema.META_DATA: {
                    'x': 'float64',
                    'y': 'float64'
                }
            }
        }

    def conf_schema(self):
        json = {
            "title": "PointNode configure",
            "type": "object",
            "properties": {
                "npts":  {
                    "type": "number",
                    "description": "number of data points",
                    "minimum": 10
                }
            },
            "required": ["npts"],
        }

        ui = {
            "npts": {"ui:widget": "updown"}
        }
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        npts = self.conf['npts']
        df = pd.DataFrame()
        df['x'] = np.random.rand(npts)
        df['y'] = np.random.rand(npts)
        output = {}
        if self.outport_connected(self.OUTPUT_PORT_NAME):
            output.update({self.OUTPUT_PORT_NAME: df})
        return output
