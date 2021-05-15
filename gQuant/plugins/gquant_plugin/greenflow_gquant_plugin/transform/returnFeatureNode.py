from greenflow.dataframe_flow import Node, PortsSpecSchema
from greenflow.dataframe_flow.portsSpecSchema import ConfSchema
from greenflow.dataframe_flow.metaSpec import MetaDataSchema
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin
from ..node_hdf_cache import NodeHDFCacheMixin

__all__ = ['ReturnFeatureNode']


class ReturnFeatureNode(TemplateNodeMixin, NodeHDFCacheMixin, Node):

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
        cols_required = {"close": "float64",
                         "asset": "int64"}
        addition = {"returns": "float64"}
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
            "title": "Add Returen Feature Node configure",
            "type": "object",
            "description": """Add the rate of of return column based
            on the `close` price for each of the asset in the dataframe.
            """,
        }
        ui = {
        }
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        Add the rate of of return column based on the `close` price for each
        of the asset in the dataframe. The result column is named as `returns`

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        tmp_col = "ae699380a8834957b3a8b7ad60192dd7"
        input_df = inputs[self.INPUT_PORT_NAME]
        shifted = input_df['close'].shift(1)
        input_df['returns'] = (input_df['close'] - shifted) / shifted
        input_df['returns'] = input_df['returns'].fillna(0.0)
        input_df[tmp_col] = (input_df['asset'] -
                             input_df['asset'].shift(1)).fillna(1)
        input_df[tmp_col] = (input_df[tmp_col] != 0).astype('int32')
        input_df[tmp_col][input_df[tmp_col] == 1] = None
        return {self.OUTPUT_PORT_NAME: input_df.dropna(
            subset=[tmp_col]).drop(tmp_col, axis=1)}
