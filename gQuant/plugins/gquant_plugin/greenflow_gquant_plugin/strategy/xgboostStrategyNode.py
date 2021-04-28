from greenflow.dataframe_flow import Node, PortsSpecSchema
from greenflow.dataframe_flow.portsSpecSchema import ConfSchema
from greenflow.dataframe_flow.metaSpec import MetaDataSchema
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin
from ..node_hdf_cache import NodeHDFCacheMixin

__all__ = ['XGBoostStrategyNode']


class XGBoostStrategyNode(TemplateNodeMixin, NodeHDFCacheMixin, Node):
    """
    This is the Node used to compute trading signal from XGBoost Strategy.

    """

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
        cols_required = {'predict': None, "asset": "int64"}
        addition = {}
        addition['signal'] = 'float64'
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
            "title": "XGBoost Node configure",
            "type": "object",
            "description": """convert the predicted next day return as trading actions
            """,
            "properties": {
            },
        }
        ui = {
        }
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        input_df = inputs[self.INPUT_PORT_NAME]
        # convert the signal to trading action
        # 1 is buy and -1 is sell
        # It predicts the tomorrow's return (shift -1)
        # We shift 1 for trading actions so that it acts on the second day
        input_df['signal'] = ((
            input_df['predict'] >= 0).astype('float') * 2 - 1).shift(1)
        # remove the bad datapints
        input_df = input_df.dropna()
        return {self.OUTPUT_PORT_NAME: input_df}
