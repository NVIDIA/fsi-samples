from greenflow.dataframe_flow import Node
from greenflow.dataframe_flow.portsSpecSchema import (PortsSpecSchema,
                                                      ConfSchema)
from greenflow.dataframe_flow.metaSpec import MetaDataSchema
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin
from ..node_hdf_cache import NodeHDFCacheMixin

__all__ = ['RenameNode']


class RenameNode(TemplateNodeMixin, NodeHDFCacheMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.INPUT_PORT_NAME = 'in'
        self.OUTPUT_PORT_NAME = 'out'
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
                port_type: "${port:in}"
            }
        }
        cols_required = {}
        meta_inports = {
            self.INPUT_PORT_NAME: cols_required
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
        retention = {}
        if 'new' in self.conf and 'old' in self.conf:
            input_meta = self.get_input_meta()
            if self.INPUT_PORT_NAME not in input_meta:
                retention = {}
            else:
                col_from_inport = input_meta[self.INPUT_PORT_NAME]
                oldType = col_from_inport[self.conf['old']]
                del col_from_inport[self.conf['old']]
                col_from_inport[self.conf['new']] = oldType
                retention = col_from_inport
        meta_outports = self.template_meta_setup().outports
        meta_outports[self.OUTPUT_PORT_NAME][MetaDataSchema.META_DATA] = \
            retention
        self.template_meta_setup(in_ports=None, out_ports=meta_outports)

    def conf_schema(self):
        json = {
            "title": "Rename Node configure",
            "type": "object",
            "description": """Rename the column name in the datafame from `old` to `new`
             defined in the node's conf""",
            "properties": {
                "old":  {
                    "type": "string",
                    "description": """the old column name that need to be
                    replaced"""
                },
                "new":  {
                    "type": "string",
                    "description": "the new column name"
                }
            },
            "required": ["old", "new"],
        }
        ui = {
        }
        input_meta = self.get_input_meta()
        if self.INPUT_PORT_NAME in input_meta:
            col_from_inport = input_meta[self.INPUT_PORT_NAME]
            enums = [col for col in col_from_inport.keys()]
            json['properties']['old']['enum'] = enums
            return ConfSchema(json=json, ui=ui)
        else:
            return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        Rename the column name in the datafame from `old` to `new` defined in
        the node's conf

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        input_df = inputs[self.INPUT_PORT_NAME]
        new_column = self.conf['new']
        old_column = self.conf['old']
        return {self.OUTPUT_PORT_NAME: input_df.rename(columns={
            old_column: new_column})}
