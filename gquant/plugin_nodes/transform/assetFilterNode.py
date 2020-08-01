from gquant.dataframe_flow import Node
from gquant.dataframe_flow._port_type_node import _PortTypesMixin
from gquant.dataframe_flow.portsSpecSchema import ConfSchema


class AssetFilterNode(Node, _PortTypesMixin):

    def init(self):
        _PortTypesMixin.init(self)
        self.INPUT_PORT_NAME = 'stock_in'
        self.OUTPUT_PORT_NAME = 'stock_out'
        cols_required = {"asset": "int64"}
        self.required = {
            self.INPUT_PORT_NAME: cols_required
        }

    def columns_setup(self):
        return _PortTypesMixin.columns_setup(self)

    def ports_setup(self):
        return _PortTypesMixin.ports_setup(self)

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
        input_df = inputs[self.INPUT_PORT_NAME]
        output_df = input_df.query('asset==%s' % self.conf["asset"])
        return {self.OUTPUT_PORT_NAME: output_df}
