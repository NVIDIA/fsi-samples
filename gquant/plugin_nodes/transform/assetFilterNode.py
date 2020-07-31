from gquant.dataframe_flow import Node
from gquant.dataframe_flow.portsSpecSchema import (PortsSpecSchema,
                                                   NodePorts,
                                                   ConfSchema)
import cudf
import dask_cudf
import pandas as pd


INPUT_PORT_NAME = 'stock_in'
OUTPUT_PORT_NAME = 'stock_out'


class AssetFilterNode(Node):

    def init(self):
        cols_required = {"asset": "int64"}
        self.required = {
            INPUT_PORT_NAME: cols_required
        }

    def columns_setup(self):
        input_columns = self.get_input_columns()
        col_from_inport = input_columns[INPUT_PORT_NAME]
        output_cols = {
            OUTPUT_PORT_NAME: col_from_inport
        }
        return output_cols

    def ports_setup(self):
        types = [cudf.DataFrame,
                 dask_cudf.DataFrame,
                 pd.DataFrame]
        port_type = PortsSpecSchema.port_type
        input_ports = {
            INPUT_PORT_NAME: {
                port_type: types
            }
        }

        output_ports = {
            OUTPUT_PORT_NAME: {
                port_type: types
            }
        }

        input_connections = self.get_connected_inports()
        if INPUT_PORT_NAME in input_connections:
            determined_type = input_connections[INPUT_PORT_NAME]
            # connected
            return NodePorts(inports={INPUT_PORT_NAME: {
                port_type: determined_type}},
                outports={OUTPUT_PORT_NAME: {
                    port_type: determined_type}})
        else:
            return NodePorts(inports=input_ports, outports=output_ports)

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
            "asset": {"ui:widget": "updown"}
        }
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
        input_df = inputs[INPUT_PORT_NAME]
        output_df = input_df.query('asset==%s' % self.conf["asset"])
        return {OUTPUT_PORT_NAME: output_df}
