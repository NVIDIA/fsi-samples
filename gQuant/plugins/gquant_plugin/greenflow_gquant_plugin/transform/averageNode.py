from greenflow.dataframe_flow import Node, PortsSpecSchema
from greenflow.dataframe_flow.portsSpecSchema import ConfSchema
from greenflow.dataframe_flow.metaSpec import MetaDataSchema
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin
from ..node_hdf_cache import NodeHDFCacheMixin

__all__ = ['AverageNode']


class AverageNode(TemplateNodeMixin, NodeHDFCacheMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.INPUT_PORT_NAME = 'stock_in'
        self.OUTPUT_PORT_NAME = 'stock_out'
        port_type = PortsSpecSchema.port_type
        port_inports = {
            self.INPUT_PORT_NAME: {
                port_type: [
                    "pandas.DataFrame", "cudf.DataFrame",
                    "dask_cudf.DataFrame", "dask.dataframe.DataFrame"
                ]
            },
        }
        port_outports = {
            self.OUTPUT_PORT_NAME: {
                port_type: "${port:stock_in}"
            }
        }
        cols_required = {"asset": "int64"}
        if 'column' in self.conf:
            retention = {
                "asset": "int64",
                self.conf['column']: "float64",
            }
        else:
            retention = {"asset": "int64"}
        meta_inports = {
            self.INPUT_PORT_NAME: cols_required
        }
        meta_outports = {
            self.OUTPUT_PORT_NAME: {
                MetaDataSchema.META_OP: MetaDataSchema.META_OP_RETENTION,
                MetaDataSchema.META_DATA: retention
            }
        }
        self.template_ports_setup(
            in_ports=port_inports,
            out_ports=port_outports
        )
        self.template_meta_setup(
            in_ports=meta_inports,
            out_ports=meta_outports
        )

    def conf_schema(self):
        input_meta = self.get_input_meta()
        json = {
            "title": "Asset Average Configure",
            "type": "object",
            "description": """Compute the average value of the key column
            which is defined in the configuration
            """,
            "properties": {
                "column":  {
                    "type": "string",
                    "description": """the column name in the dataframe
                    to average"""
                }
            },
            "required": ["column"],
        }
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
        Compute the average value of the key column which is defined in the
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
        average_column = self.conf['column']
        volume_df = input_df[[average_column, "asset"]] \
            .groupby(["asset"]).mean().reset_index()
        volume_df.columns = ['asset', average_column]
        return {self.OUTPUT_PORT_NAME: volume_df}
