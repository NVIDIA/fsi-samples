from greenflow.dataframe_flow import Node, PortsSpecSchema
from .._port_type_node import _PortTypesMixin
from greenflow.dataframe_flow.portsSpecSchema import ConfSchema


class DropNode(_PortTypesMixin, Node):

    def init(self):
        _PortTypesMixin.init(self)
        port_type = PortsSpecSchema.port_type
        self.INPUT_PORT_NAME = 'in'
        self.OUTPUT_PORT_NAME = 'out'
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
                port_type: "${port:in}"
            }
        }
        self.meta_inports = {
            self.INPUT_PORT_NAME: {}
        }
        dropped = {}
        for k in self.conf.get('columns', {}):
            dropped[k] = None
        self.meta_outports = {
            self.OUTPUT_PORT_NAME: {
                self.META_OP: self.META_OP_DELETION,
                self.META_REF_INPUT: self.INPUT_PORT_NAME,
                self.META_DATA: dropped
            }
        }

    def meta_setup(self):
        return _PortTypesMixin.meta_setup(self)

    def ports_setup(self):
        return _PortTypesMixin.ports_setup(self)

    def conf_schema(self):
        json = {
            "title": "Drop Column configure",
            "type": "object",
            "description": """Drop a few columns from the dataframe""",
            "properties": {
                "columns":  {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": """array of columns to be droped"""
                }
            },
            "required": ["columns"],
        }
        ui = {
            "columns": {
                "items": {
                    "ui:widget": "text"
                }
            },
        }
        input_meta = self.get_input_meta()
        if self.INPUT_PORT_NAME in input_meta:
            col_from_inport = input_meta[self.INPUT_PORT_NAME]
            enums = [col for col in col_from_inport.keys()]
            json['properties']['columns']['items']['enum'] = enums
            ui = {}
            return ConfSchema(json=json, ui=ui)
        else:
            ui = {
                "column": {"ui:widget": "text"}
            }
            return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        Drop a few columns from the dataframe that are defined in the `columns`
        in the nodes' conf

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        input_df = inputs[self.INPUT_PORT_NAME]
        column_names = self.conf['columns']
        return {self.OUTPUT_PORT_NAME: input_df.drop(column_names, axis=1)}
