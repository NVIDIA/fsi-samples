from greenflow.dataframe_flow import Node
from greenflow.dataframe_flow.portsSpecSchema import ConfSchema
from greenflow.dataframe_flow.portsSpecSchema import (PortsSpecSchema,
                                                   MetaData,
                                                   NodePorts)
import cudf
import dask_cudf
import pandas as pd
from .._port_type_node import _PortTypesMixin


class LeftMergeNode(_PortTypesMixin, Node):

    def init(self):
        self.INPUT_PORT_LEFT_NAME = 'left'
        self.INPUT_PORT_RIGHT_NAME = 'right'
        self.OUTPUT_PORT_NAME = 'merged'

    def ports_setup_from_types(self, types):
        port_type = PortsSpecSchema.port_type
        input_ports = {
            self.INPUT_PORT_LEFT_NAME: {
                port_type: types
            },
            self.INPUT_PORT_RIGHT_NAME: {
                port_type: types
            }
        }

        output_ports = {
            self.OUTPUT_PORT_NAME: {
                port_type: types
            }
        }

        input_connections = self.get_connected_inports()
        if (self.INPUT_PORT_LEFT_NAME in input_connections and
                self.INPUT_PORT_RIGHT_NAME in input_connections):
            determined_type1 = input_connections[self.INPUT_PORT_LEFT_NAME]
            determined_type2 = input_connections[self.INPUT_PORT_RIGHT_NAME]
            if (determined_type1 == determined_type2):
                # connected
                return NodePorts(inports={self.INPUT_PORT_LEFT_NAME: {
                    port_type: determined_type1},
                    self.INPUT_PORT_RIGHT_NAME: {
                    port_type: determined_type1}},
                    outports={self.OUTPUT_PORT_NAME: {
                        port_type: determined_type1}})
            else:
                return NodePorts(inports=input_ports, outports=output_ports)
        else:
            return NodePorts(inports=input_ports, outports=output_ports)

    def ports_setup(self):
        types = [cudf.DataFrame,
                 dask_cudf.DataFrame,
                 pd.DataFrame]
        return self.ports_setup_from_types(types)

    def meta_setup(self):
        cols_required = {}
        required = {
            self.INPUT_PORT_LEFT_NAME: cols_required,
            self.INPUT_PORT_RIGHT_NAME: cols_required
        }
        input_meta = self.get_input_meta()
        if (self.INPUT_PORT_LEFT_NAME in input_meta
                and self.INPUT_PORT_RIGHT_NAME in input_meta):
            col_from_left_inport = input_meta[self.INPUT_PORT_LEFT_NAME]
            col_from_right_inport = input_meta[self.INPUT_PORT_RIGHT_NAME]
            col_from_left_inport.update(col_from_right_inport)
            output_cols = {
                self.OUTPUT_PORT_NAME: col_from_left_inport
            }
            metadata = MetaData(inports=required, outports=output_cols)
            return metadata
        elif self.INPUT_PORT_LEFT_NAME in input_meta:
            col_from_left_inport = input_meta[self.INPUT_PORT_LEFT_NAME]
            output_cols = {
                self.OUTPUT_PORT_NAME: col_from_left_inport
            }
            metadata = MetaData(inports=required, outports=output_cols)
            return metadata
        elif self.INPUT_PORT_RIGHT_NAME in input_meta:
            col_from_right_inport = input_meta[self.INPUT_PORT_RIGHT_NAME]
            output_cols = {
                self.OUTPUT_PORT_NAME: col_from_right_inport
            }
            metadata = MetaData(inports=required, outports=output_cols)
            return metadata
        else:
            metadata = MetaData(inports=required,
                                outports={self.OUTPUT_PORT_NAME: {}})
            return metadata

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
