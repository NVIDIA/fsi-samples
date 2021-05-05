from greenflow.dataframe_flow import (
    Node, PortsSpecSchema, ConfSchema, MetaDataSchema)
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin
from ..node_hdf_cache import NodeHDFCacheMixin


class SimpleBackTestNode(TemplateNodeMixin, NodeHDFCacheMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.INPUT_PORT_NAME = 'bardata_in'
        self.OUTPUT_PORT_NAME = 'backtest_out'
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
                port_type: "${port:bardata_in}"
            }
        }
        cols_required = {"signal": "float64",
                         "returns": "float64"}
        addition = {"strategy_returns": "float64"}
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
            "title": "Backtest configure",
            "type": "object",
            "description": """compute the `strategy_returns` by assuming invest
             `signal` amount of dollars for each of the time step.""",
        }
        ui = {
        }
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        compute the `strategy_returns` by assuming invest `signal` amount of
        dollars for each of the time step.

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        input_df = inputs[self.INPUT_PORT_NAME]
        input_df['strategy_returns'] = input_df['signal'] * input_df['returns']
        return {self.OUTPUT_PORT_NAME: input_df}
