from gquant.dataframe_flow import Node
from gquant.dataframe_flow._port_type_node import _PortTypesMixin
from gquant.dataframe_flow.portsSpecSchema import (ConfSchema,
                                                   PortsSpecSchema, NodePorts)
from .data_obj import ProjectionData 
import cupy as cp
import copy
from collections import OrderedDict

__all__ = ['LinearEmbeddingNode']

SPECIAL_OUTPUT_DIM_COL = 'OUTPUT_DIM_23b1c5ce-e0bf-11ea-afcf-80e82cc76d44'


class LinearEmbeddingNode(Node, _PortTypesMixin):

    def init(self):
        _PortTypesMixin.init(self)
        self.INPUT_PORT_NAME = 'df_in'
        self.OUTPUT_PORT_NAME = 'df_out'
        self.INPUT_PROJ_NAME = 'proj_data_in'
        self.OUTPUT_PROJ_NAME = 'proj_data_out'
        cols_required = {}
        self.required = {
            self.INPUT_PORT_NAME: cols_required,
            self.INPUT_PROJ_NAME: cols_required
        }

    def ports_setup_from_types(self, types):
        port_type = PortsSpecSchema.port_type
        input_ports = {
            self.INPUT_PORT_NAME: {
                port_type: types
            },
            self.INPUT_PROJ_NAME: {
                port_type: ProjectionData
            }
        }

        output_ports = {
            self.OUTPUT_PORT_NAME: {
                port_type: types
            },
            self.OUTPUT_PROJ_NAME: {
                port_type: ProjectionData
            }
        }

        input_connections = self.get_connected_inports()
        if self.INPUT_PORT_NAME in input_connections:
            determined_type = input_connections[self.INPUT_PORT_NAME]
            input_ports.update({self.INPUT_PORT_NAME:
                                {port_type: determined_type}})
            output_ports.update({self.OUTPUT_PORT_NAME: {
                                 port_type: determined_type}})
            # connected
            return NodePorts(inports=input_ports,
                             outports=output_ports)
        else:
            return NodePorts(inports=input_ports, outports=output_ports)

    def columns_setup(self):
        self.required = {
            self.INPUT_PORT_NAME: {},
            self.INPUT_PROJ_NAME: {}
        }
        if 'columns' in self.conf and self.conf.get('include', True):
            cols_required = {}
            for col in self.conf['columns']:
                cols_required[col] = None
            self.required = {
                self.INPUT_PORT_NAME: cols_required,
                self.INPUT_PROJ_NAME: cols_required
            }
        output_cols = {
            self.OUTPUT_PORT_NAME: self.required[self.INPUT_PORT_NAME],
            self.OUTPUT_PROJ_NAME: self.required[
                self.INPUT_PROJ_NAME]
        }
        input_columns = self.get_input_columns()
        if (self.INPUT_PROJ_NAME in input_columns and
                self.INPUT_PORT_NAME in input_columns):
            cols_required = copy.copy(input_columns[self.INPUT_PROJ_NAME])
            self.required = {
                self.INPUT_PORT_NAME: cols_required,
                self.INPUT_PROJ_NAME: cols_required
            }
            col_from_inport = input_columns[self.INPUT_PORT_NAME]
            if SPECIAL_OUTPUT_DIM_COL in cols_required:
                out_dim = cols_required[SPECIAL_OUTPUT_DIM_COL]
                del cols_required[SPECIAL_OUTPUT_DIM_COL]
                cols = ['em'+str(i) for i in range(out_dim)]
                for col in cols:
                    col_from_inport[col] = None
            output_cols = {
                self.OUTPUT_PORT_NAME: col_from_inport,
                self.OUTPUT_PROJ_NAME: cols_required
            }
            return output_cols
        elif (self.INPUT_PROJ_NAME in input_columns and
              self.INPUT_PORT_NAME not in input_columns):
            cols_required = copy.copy(input_columns[self.INPUT_PROJ_NAME])
            self.required = {
                self.INPUT_PORT_NAME: cols_required,
                self.INPUT_PROJ_NAME: cols_required
            }
            output = copy.copy(cols_required) 
            if SPECIAL_OUTPUT_DIM_COL in cols_required:
                out_dim = cols_required[SPECIAL_OUTPUT_DIM_COL]
                del cols_required[SPECIAL_OUTPUT_DIM_COL]
                cols = ['em'+str(i) for i in range(out_dim)]
                for col in cols:
                    output[col] = None
            output_cols = {
                self.OUTPUT_PORT_NAME: output,
                self.OUTPUT_PROJ_NAME: cols_required
            }
            return output_cols
        elif (self.INPUT_PROJ_NAME not in input_columns and
              self.INPUT_PORT_NAME in input_columns):
            col_from_inport = input_columns[self.INPUT_PORT_NAME]
            enums = [col for col in col_from_inport.keys()]
            if 'columns' in self.conf:
                if self.conf.get('include', True):
                    included_colums = self.conf['columns']
                else:
                    included_colums = [col for col in enums
                                       if col not in self.conf['columns']]
                cols_required = OrderedDict()
                for col in included_colums:
                    if col in col_from_inport:
                        cols_required[col] = col_from_inport[col]
                    else:
                        cols_required[col] = None
                self.required = {
                    self.INPUT_PORT_NAME: cols_required,
                    self.INPUT_PROJ_NAME: cols_required
                }
                col_dict = ['em'+str(i) for i in range(
                    self.conf['out_dimension'])]
                for col in col_dict:
                    col_from_inport[col] = None
                proj_out = copy.copy(cols_required)
                proj_out[SPECIAL_OUTPUT_DIM_COL] = self.conf['out_dimension']
                output_cols = {
                    self.OUTPUT_PORT_NAME: col_from_inport,
                    self.OUTPUT_PROJ_NAME: proj_out
                }
            return output_cols
        return output_cols

    def ports_setup(self):
        return _PortTypesMixin.ports_setup(self)

    def conf_schema(self):
        json = {
            "title": "Linear Embeding configure",
            "type": "object",
            "description": """Project the features randomly and linearly to a
            space of different dimension. It generates the random projection
            matrix of size feature_dim x out_dimension and does dot product
            with the input dataframe""",
            "properties": {
                "columns":  {
                    "type": "array",
                    "description": """an array of columns that need to
                     be normalized, or excluded from normalization depending
                     on the `incldue` flag state""",
                    "items": {
                        "type": "string"
                    }
                },
                "include":  {
                    "type": "boolean",
                    "description": """if set true, the `columns` need to be 
                    normalized. if false, all dataframe columns except the
                    `columns` need to be normalized""",
                    "default": True
                },
                "out_dimension": {
                    "type": "integer",
                    "minimum": 0,
                    "description": """the projected dimension size"""
                },
                "seed": {
                    "type": "integer",
                    "description": """the seed number for random projection"""
                }
            },
            "required": [],
        }
        ui = {}
        input_columns = self.get_input_columns()
        if self.INPUT_PORT_NAME in input_columns:
            col_from_inport = input_columns[self.INPUT_PORT_NAME]
            enums = [col for col in col_from_inport.keys()]
            json['properties']['columns']['items']['enum'] = enums
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        normalize the data to zero mean, std 1

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        input_df = inputs[self.INPUT_PORT_NAME]

        if self.INPUT_PROJ_NAME in inputs:
            data_in = inputs[self.INPUT_PROJ_NAME].data
            input_columns = self.get_input_columns()
            col_from_inport = input_columns[self.INPUT_PROJ_NAME]
            proj_data = data_in
            cols = []
            # it has  required colmns that used to do mapping
            for col in col_from_inport.keys():
                if col != SPECIAL_OUTPUT_DIM_COL:
                    cols.append(col)
        else:
            if self.conf.get('include', True):
                cols = self.conf['columns']
            else:
                cols = input_df.columns.difference(
                    self.conf['columns']).values.tolist()
            # need to generate the random projection
            if 'seed' in self.conf:
                cp.random.seed(self.conf['seed'])
            proj_data = cp.random.rand(len(cols), self.conf['out_dimension'])
        cols.sort()
        # print(self.uid, cols)
        # print(self.uid, proj_data)
        output_matrix = input_df[cols].values.dot(proj_data)
        col_dict = {'em'+str(i): output_matrix[:, i]
                    for i in range(proj_data.shape[1])}
        # output_df = input_df[input_df.columns.difference(cols)]
        output_df = input_df.assign(**col_dict)
        output = {}
        if self.outport_connected(self.OUTPUT_PORT_NAME):
            output.update({self.OUTPUT_PORT_NAME: output_df})
        if self.outport_connected(self.OUTPUT_PROJ_NAME):
            payload = ProjectionData(proj_data)
            output.update({self.OUTPUT_PROJ_NAME: payload})
        return output
