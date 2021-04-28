from greenflow.dataframe_flow import Node, PortsSpecSchema
from greenflow.dataframe_flow.portsSpecSchema import ConfSchema
from greenflow.dataframe_flow.metaSpec import MetaDataSchema
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin
from ..node_hdf_cache import NodeHDFCacheMixin

__all__ = ['AssetIndicatorNode']


class AssetIndicatorNode(TemplateNodeMixin, NodeHDFCacheMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.delayed_process = True
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
        addition = {"indicator": "int32"}
        meta_inports = {
            self.INPUT_PORT_NAME: cols_required
        }
        meta_outports = {
            self.OUTPUT_PORT_NAME: {
                MetaDataSchema.META_OP: MetaDataSchema.META_OP_ADDITION,
                MetaDataSchema.META_REF_INPUT: self.INPUT_PORT_NAME,
                MetaDataSchema.META_DATA: addition
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
        json = {
            "title": "Asset indicator configure",
            "type": "object",
            "description": """Add the indicator column in the dataframe which
             set 1 at the beginning of the each of the assets, assuming the
             rows are sorted so same asset are grouped together""",
        }
        ui = {
        }
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        Add the indicator column in the dataframe which set 1 at the beginning
        of the each of the assets, assuming the rows are sorted so same asset
        are grouped together

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """

        input_df = inputs[self.INPUT_PORT_NAME]
        input_df['indicator'] = (input_df['asset'] -
                                 input_df['asset'].shift(1)).fillna(1)
        input_df['indicator'] = (input_df['indicator'] != 0).astype('int32')
        return {self.OUTPUT_PORT_NAME: input_df}
