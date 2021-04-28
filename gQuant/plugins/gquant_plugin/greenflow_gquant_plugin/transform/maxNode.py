from greenflow.dataframe_flow import (
    Node, ConfSchema, PortsSpecSchema, MetaDataSchema)
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin
from ..node_hdf_cache import NodeHDFCacheMixin

__all__ = ['MaxNode']


class MaxNode(TemplateNodeMixin, NodeHDFCacheMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.INPUT_PORT_NAME = 'in'
        self.OUTPUT_PORT_NAME = 'out'
        port_type = PortsSpecSchema.port_type
        self.port_inports = {
            self.INPUT_PORT_NAME: {
                port_type: [
                    "pandas.DataFrame", "cudf.DataFrame",
                    "dask_cudf.DataFrame", "dask.dataframe.DataFrame"
                ]
            },
        }
        self.port_outports = {
            self.OUTPUT_PORT_NAME: {
                port_type: "${port:in}"
            }
        }
        cols_required = {"asset": "int64"}
        if 'column' in self.conf:
            retention = {self.conf['column']: "float64",
                         "asset": "int64"}
        else:
            retention = {"asset": "int64"}
        self.meta_inports = {
            self.INPUT_PORT_NAME: cols_required
        }
        self.meta_outports = {
            self.OUTPUT_PORT_NAME: {
                MetaDataSchema.META_OP: MetaDataSchema.META_OP_RETENTION,
                MetaDataSchema.META_DATA: retention
            }
        }

    def conf_schema(self):
        json = {
            "title": "Maximum Value Node configure",
            "type": "object",
            "description": "Compute the maximum value of the key column",
            "properties": {
                "column":  {
                    "type": "string",
                    "description": "column to calculate the maximum value"
                }
            },
            "required": ["column"],
        }
        input_meta = self.get_input_meta()
        if self.INPUT_PORT_NAME in input_meta:
            col_from_inport = input_meta[self.INPUT_PORT_NAME]
            enums = [col for col in col_from_inport.keys()]
            json['properties']['column']['enum'] = enums
            ui = {}
            return ConfSchema(json=json, ui=ui)
        else:
            ui = {
                "column": {"ui:widget": "text"}
            }
            return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        Compute the maximum value of the key column which is defined in the
        `column` of the node's conf

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        input_df = inputs[self.INPUT_PORT_NAME]
        max_column = self.conf['column']
        volume_df = input_df[[max_column,
                              "asset"]].groupby(["asset"]).max().reset_index()
        volume_df.columns = ['asset', max_column]
        return {self.OUTPUT_PORT_NAME: volume_df}
