from greenflow.dataframe_flow import Node
from greenflow.dataframe_flow.portsSpecSchema import ConfSchema
from greenflow.dataframe_flow.portsSpecSchema import PortsSpecSchema
from greenflow.dataframe_flow.metaSpec import MetaDataSchema
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin
from ..node_hdf_cache import NodeHDFCacheMixin

__all__ = ['LeftMergeNode']


class LeftMergeNode(TemplateNodeMixin, NodeHDFCacheMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.INPUT_PORT_LEFT_NAME = 'left'
        self.INPUT_PORT_RIGHT_NAME = 'right'
        self.OUTPUT_PORT_NAME = 'merged'
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
            },
        }
        port_outports = {
            self.OUTPUT_PORT_NAME: {
                port_type: "${port:left}"
            }
        }
        cols_required = {}
        meta_inports = {
            self.INPUT_PORT_LEFT_NAME: cols_required,
            self.INPUT_PORT_RIGHT_NAME: cols_required
        }
        meta_outports = {
            self.OUTPUT_PORT_NAME: {
                MetaDataSchema.META_OP: MetaDataSchema.META_OP_RETENTION,
                MetaDataSchema.META_DATA: {}
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

    def update(self):
        TemplateNodeMixin.update(self)
        input_meta = self.get_input_meta()
        output_cols = {}
        if (self.INPUT_PORT_LEFT_NAME in input_meta
                and self.INPUT_PORT_RIGHT_NAME in input_meta):
            col_from_left_inport = input_meta[self.INPUT_PORT_LEFT_NAME]
            col_from_right_inport = input_meta[self.INPUT_PORT_RIGHT_NAME]
            col_from_left_inport.update(col_from_right_inport)
            output_cols = col_from_left_inport
        elif self.INPUT_PORT_LEFT_NAME in input_meta:
            col_from_left_inport = input_meta[self.INPUT_PORT_LEFT_NAME]
            output_cols = col_from_left_inport
        elif self.INPUT_PORT_RIGHT_NAME in input_meta:
            col_from_right_inport = input_meta[self.INPUT_PORT_RIGHT_NAME]
            output_cols = col_from_right_inport
        meta_outports = self.template_meta_setup().outports
        meta_outports[self.OUTPUT_PORT_NAME][MetaDataSchema.META_DATA] = \
            output_cols
        self.template_meta_setup(
            in_ports=None,
            out_ports=meta_outports
        )

    def conf_schema(self):
        json = {
            "title": "DataFrame Left Merge configure",
            "type": "object",
            "description": """Left merge two dataframes of the same types""",
            "properties": {
                "column":  {
                    "type": "string",
                    "description": "column name on which to do the left merge"
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
        return {self.OUTPUT_PORT_NAME: df1.merge(df2, on=self.conf['column'],
                                                 how='left')}
