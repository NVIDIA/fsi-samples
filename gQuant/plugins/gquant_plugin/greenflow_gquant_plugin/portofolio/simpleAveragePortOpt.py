from greenflow.dataframe_flow import Node, PortsSpecSchema
from greenflow.dataframe_flow.portsSpecSchema import ConfSchema
from greenflow.dataframe_flow.metaSpec import MetaDataSchema
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin
from ..node_hdf_cache import NodeHDFCacheMixin


class SimpleAveragePortOpt(TemplateNodeMixin, NodeHDFCacheMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.INPUT_PORT_NAME = 'stock_in'
        self.OUTPUT_PORT_NAME = 'stock_out'
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
                port_type: "${port:stock_in}"
            }
        }
        cols_required = {"datetime": "datetime64[ns]",
                         "strategy_returns": "float64",
                         "asset": "int64"}
        retention = {"datetime": "datetime64[ns]",
                     "strategy_returns": "float64"}
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
            "title": "Simple Portfolio Node configure",
            "type": "object",
            "description": """Average the strategy returns for all the
            assets """,
        }
        ui = {
        }
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        Average the strategy returns for all the assets.

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        input_df = inputs[self.INPUT_PORT_NAME]
        port = input_df[['datetime', 'strategy_returns']] \
            .groupby(['datetime']).mean().reset_index().sort_values('datetime')
        port.columns = ['datetime', 'strategy_returns']
        return {self.OUTPUT_PORT_NAME: port}
