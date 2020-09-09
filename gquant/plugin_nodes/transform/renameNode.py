from gquant.dataframe_flow import Node
from gquant.dataframe_flow.portsSpecSchema import ConfSchema
from gquant.dataframe_flow._port_type_node import _PortTypesMixin


class RenameNode(Node, _PortTypesMixin):

    def init(self):
        _PortTypesMixin.init(self)
        self.INPUT_PORT_NAME = 'in'
        self.OUTPUT_PORT_NAME = 'out'
        cols_required = {}
        self.required = {
            self.INPUT_PORT_NAME: cols_required
        }

    def ports_setup(self):
        return _PortTypesMixin.ports_setup(self)

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
        input_columns = self.get_input_columns()
        if self.INPUT_PORT_NAME in input_columns:
            col_from_inport = input_columns[self.INPUT_PORT_NAME]
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

    def columns_setup(self):
        if 'new' in self.conf and 'old' in self.conf:
            input_columns = self.get_input_columns()
            if self.INPUT_PORT_NAME not in input_columns:
                return {self.OUTPUT_PORT_NAME: {}}
            else:
                col_from_inport = input_columns[self.INPUT_PORT_NAME]
                oldType = col_from_inport[self.conf['old']]
                del col_from_inport[self.conf['old']]
                col_from_inport[self.conf['new']] = oldType
                return _PortTypesMixin.retention_columns_setup(self,
                                                               col_from_inport)
        else:
            return {self.OUTPUT_PORT_NAME: {}}
