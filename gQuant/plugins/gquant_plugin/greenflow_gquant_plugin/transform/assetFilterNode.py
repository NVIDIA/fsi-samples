from greenflow.dataframe_flow import Node
from greenflow.dataframe_flow.portsSpecSchema import (ConfSchema,
                                                      PortsSpecSchema)
from greenflow.dataframe_flow.metaSpec import MetaDataSchema
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin
from ..node_hdf_cache import NodeHDFCacheMixin

__all__ = ["AssetFilterNode"]


class AssetFilterNode(TemplateNodeMixin, NodeHDFCacheMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.INPUT_PORT_NAME = 'stock_in'
        self.OUTPUT_PORT_NAME = 'stock_out'
        self.INPUT_MAP_NAME = 'name_map'
        self.OUTPUT_ASSET_NAME = 'stock_name'

        port_type = PortsSpecSchema.port_type
        port_inports = {
            self.INPUT_PORT_NAME: {
                port_type: [
                    "pandas.DataFrame", "cudf.DataFrame",
                    "dask_cudf.DataFrame", "dask.dataframe.DataFrame"
                ]
            },
            self.INPUT_MAP_NAME: {
                port_type: [
                    "greenflow_gquant_plugin.dataloader.stockMap.StockMap"
                ]
            }
        }
        port_outports = {
            self.OUTPUT_PORT_NAME: {
                port_type: "${port:stock_in}"
            },
            self.OUTPUT_ASSET_NAME: {
                port_type: ['builtins.str']
            }
        }
        cols_required = {"asset": "int64"}
        meta_inports = {
            self.INPUT_PORT_NAME: cols_required,
            self.INPUT_MAP_NAME: {}
        }
        meta_outports = {
            self.OUTPUT_PORT_NAME: {
                MetaDataSchema.META_OP: MetaDataSchema.META_OP_ADDITION,
                MetaDataSchema.META_REF_INPUT: self.INPUT_PORT_NAME,
                MetaDataSchema.META_DATA: {}
            },
            self.OUTPUT_ASSET_NAME: {
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
        name = self._find_asset_name()
        asset_retension = {"asset_name": name}
        meta_outports = self.template_meta_setup().outports
        meta_outports[self.OUTPUT_ASSET_NAME][
            MetaDataSchema.META_DATA] = asset_retension
        self.template_meta_setup(
            in_ports=None,
            out_ports=meta_outports
        )

    def _find_asset_name(self):
        name = ""
        input_meta = self.get_input_meta()
        if self.outport_connected(self.OUTPUT_ASSET_NAME):
            if self.INPUT_MAP_NAME in input_meta and 'asset' in self.conf:
                col_from_inport = input_meta[self.INPUT_MAP_NAME]
                enums = col_from_inport['asset']
                enumNames = col_from_inport['asset_name']
                found = False
                for i, name in zip(enums, enumNames):
                    if i == self.conf['asset']:
                        found = True
                        break
                if not found:
                    name = ""
        return name

    def conf_schema(self):
        json = {
            "title": "Asset Filter Node configure",
            "type": "object",
            "description": "select the asset based on asset id",
            "properties": {
                "asset":  {
                    "type": "number",
                    "description": "asset id number"
                }
            },
            "required": ["asset"],
        }
        ui = {
        }
        input_meta = self.get_input_meta()
        if self.INPUT_MAP_NAME in input_meta:
            col_from_inport = input_meta[self.INPUT_MAP_NAME]
            enums = col_from_inport['asset']
            enumNames = col_from_inport['asset_name']
            json['properties']['asset']['enum'] = enums
            json['properties']['asset']['enumNames'] = enumNames
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        select the asset based on asset id, which is defined in `asset` in the
        nodes' conf

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        input_df = inputs[self.INPUT_PORT_NAME]
        output_df = input_df.query('asset==%s' % self.conf["asset"])
        output = {self.OUTPUT_PORT_NAME: output_df}
        if self.outport_connected(self.OUTPUT_ASSET_NAME):
            name = self._find_asset_name()
            output.update({self.OUTPUT_ASSET_NAME: name})
        return output
