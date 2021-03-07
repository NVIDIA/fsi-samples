from greenflow.dataframe_flow import Node
from greenflow.dataframe_flow.portsSpecSchema import ConfSchema
from greenflow.dataframe_flow.portsSpecSchema import PortsSpecSchema
from .._port_type_node import _PortTypesMixin


class LeftMergeNode(_PortTypesMixin, Node):

    def init(self):
        _PortTypesMixin.init(self)
        self.INPUT_PORT_LEFT_NAME = 'left'
        self.INPUT_PORT_RIGHT_NAME = 'right'
        self.OUTPUT_PORT_NAME = 'merged'
        port_type = PortsSpecSchema.port_type
        self.port_inports = {
            self.INPUT_PORT_LEFT_NAME: {
                port_type: [
                    "pandas.DataFrame", "cudf.DataFrame",
                    "dask_cudf.DataFrame", "dask.dataframe.DataFrame"
                ]
            },
            self.INPUT_PORT_RIGHT_NAME: {
                port_type: [
                    "pandas.DataFrame", "cudf.DataFrame",
                    "dask_cudf.DataFrame", "dask.dataframe.DataFrame"
                ]
            },
        }
        self.port_outports = {
            self.OUTPUT_PORT_NAME: {
                port_type: "${port:left}"
            }
        }
        cols_required = {}
        self.meta_inports = {
            self.INPUT_PORT_LEFT_NAME: cols_required,
            self.INPUT_PORT_RIGHT_NAME: cols_required
        }

    def update(self):
        input_meta = self.get_input_meta()
        output_cols = {}
        if (self.INPUT_PORT_LEFT_NAME in input_meta
                and self.INPUT_PORT_RIGHT_NAME in input_meta):
            col_from_left_inport = input_meta[self.INPUT_PORT_LEFT_NAME]
            col_from_right_inport = input_meta[self.INPUT_PORT_RIGHT_NAME]
            col_from_left_inport.update(col_from_right_inport)
            output_cols = col_from_left_inport
        elif self.INPUT_PORT_LEFT_NAME in input_meta:
            col_from_left_inport = input_meta[self.INPUT_PORT_LEFT_NAME]
            output_cols = col_from_left_inport
        elif self.INPUT_PORT_RIGHT_NAME in input_meta:
            col_from_right_inport = input_meta[self.INPUT_PORT_RIGHT_NAME]
            output_cols = col_from_right_inport

        self.meta_outports = {
            self.OUTPUT_PORT_NAME: {
                self.META_OP: self.META_OP_RETENTION,
                self.META_DATA: output_cols
            }
        }
        _PortTypesMixin.update(self)

    def ports_setup(self):
        return _PortTypesMixin.ports_setup(self)

    def meta_setup(self):
        return _PortTypesMixin.meta_setup(self)

    def conf_schema(self):
        json = {
            "title": "DataFrame Left Merge configure",
            "type": "object",
            "description": """Left merge two dataframes of the same types""",
            "properties": {
                "column":  {
                    "type": "string",
                    "description": "column name on which to do the left merge"
                }
            },
            "required": ["column"],
        }
        input_meta = self.get_input_meta()
        if (self.INPUT_PORT_LEFT_NAME in input_meta
                and self.INPUT_PORT_RIGHT_NAME in input_meta):
            col_left_inport = input_meta[self.INPUT_PORT_LEFT_NAME]
            col_right_inport = input_meta[self.INPUT_PORT_RIGHT_NAME]
            enums1 = set([col for col in col_left_inport.keys()])
            enums2 = set([col for col in col_right_inport.keys()])
            json['properties']['column']['enum'] = list(
                enums1.intersection(enums2))
            ui = {}
            return ConfSchema(json=json, ui=ui)
        else:
            ui = {
                "column": {"ui:widget": "text"}
            }
            return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        left merge the two dataframes in the inputs. the `on column` is defined
        in the `column` of the node's conf

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        df1 = inputs[self.INPUT_PORT_LEFT_NAME]
        df2 = inputs[self.INPUT_PORT_RIGHT_NAME]
        return {self.OUTPUT_PORT_NAME: df1.merge(df2, on=self.conf['column'],
                                                 how='left')}
