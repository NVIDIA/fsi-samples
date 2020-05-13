import gquant
from gquant.dataframe_flow.portsSpecSchema import ConfSchema
import json

data = """{
  "conf": {
    "input": [
      "train_norm.df_in",
      "test_norm.df_in"
    ],
    "output": [
      "train_infer.out",
      "test_infer.out",
      "train_xgboost.model_out",
      "train_norm.df_out",
      "test_norm.df_out"
    ],
    "subnode_ids": [
      "train_norm",
      "train_xgboost"
    ],
    "subnodes_conf": {},
    "taskgraph": "taskgraphs/xgboost_example/xgboost_model.gq.yaml"
  }
}
"""


class CustXGBoostNode(gquant.plugin_nodes.util.CompositeNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # modify the self.conf to the one that this Composite node wants
        global data
        node_conf = self.conf
        data_obj = json.loads(data)
        data_obj['conf']['subnodes_conf'].update(node_conf)
        self.conf = data_obj['conf']

    def conf_schema(self):
        full_schema = super().conf_schema()
        full_schema_json = full_schema.json
        ui = full_schema.ui
        json = {
            "title": "CustXGBoostNode configure",
            "type": "object",
            "description": "Enter your node description here",
            "properties": {
            }
        }
        item_dict = full_schema_json['properties']["subnodes_conf"]['properties']
        for key in item_dict.keys():
            json['properties'][key] = item_dict[key]
        return ConfSchema(json=json, ui=ui)

      