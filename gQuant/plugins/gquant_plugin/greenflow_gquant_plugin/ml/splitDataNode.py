from .._port_type_node import _PortTypesMixin
from greenflow.dataframe_flow.portsSpecSchema import (ConfSchema, MetaData,
                                                   NodePorts, PortsSpecSchema)
from greenflow.dataframe_flow import Node
import cudf
import dask_cudf
import cuml
import copy


__all__ = ['DataSplittingNode']


class DataSplittingNode(_PortTypesMixin, Node):

    def init(self):
        self.delayed_process = True
        self.INPUT_PORT_NAME = 'in'
        self.OUTPUT_PORT_NAME_TRAIN = 'train'
        self.OUTPUT_PORT_NAME_TEST = 'test'

    def ports_setup_from_types(self, types):
        port_type = PortsSpecSchema.port_type
        input_ports = {
            self.INPUT_PORT_NAME: {
                port_type: types
            }
        }

        output_ports = {
            self.OUTPUT_PORT_NAME_TRAIN: {
                port_type: types
            },
            self.OUTPUT_PORT_NAME_TEST: {
                port_type: types
            }
        }
        input_connections = self.get_connected_inports()
        if self.INPUT_PORT_NAME in input_connections:
            determined_type = input_connections[self.INPUT_PORT_NAME]
            # connected
            return NodePorts(inports={self.INPUT_PORT_NAME: {
                port_type: determined_type}},
                outports={self.OUTPUT_PORT_NAME_TEST: {
                    port_type: determined_type},
                    self.OUTPUT_PORT_NAME_TRAIN: {
                    port_type: determined_type}
            })
        else:
            return NodePorts(inports=input_ports, outports=output_ports)

    def ports_setup(self):
        types = [cudf.DataFrame,
                 dask_cudf.DataFrame]
        return self.ports_setup_from_types(types)

    def meta_setup(self):
        cols_required = {}
        col_from_inport = {}
        required = {
            self.INPUT_PORT_NAME: cols_required
        }
        input_meta = self.get_input_meta()
        if self.INPUT_PORT_NAME in input_meta:
            col_inport = input_meta[self.INPUT_PORT_NAME]
            if 'target' in self.conf:
                target_col = self.conf['target']
                for i in sorted(col_inport.keys()):
                    if i != target_col:
                        col_from_inport[i] = col_inport[i]
                col_from_inport[target_col] = col_inport[target_col]
                if target_col in col_inport:
                    required[self.INPUT_PORT_NAME][target_col] = \
                        col_inport[target_col]
        else:
            col_from_inport = required[self.INPUT_PORT_NAME]
        output_cols = {
            self.OUTPUT_PORT_NAME_TRAIN: col_from_inport,
            self.OUTPUT_PORT_NAME_TEST: col_from_inport
        }
        metadata = MetaData(inports=required, outports=output_cols)
        return metadata

    def conf_schema(self):
        json = {
            "title": "Data Splitting configure",
            "type": "object",
            "description": """Partitions device data into two parts""",
            "properties": {
                "target": {"type": "string",
                           "description": "Target column name"},
                "train_size": {"type": "number",
                               "description": """If float, represents the
                               proportion [0, 1] of the data to be assigned to
                               the training set. If an int, represents the
                               number of instances to be assigned to the
                               training set.""",
                               "default": 0.8},
                "shuffle": {"type": "boolean",
                            "description": """Whether or not to shuffle inputs
                            before splitting random_stateint"""},
                "random_state": {"type": "number",
                                 "description": """If shuffle is true, seeds
                                 the generator. Unseeded by default"""}
            },
            "required": ["target"],
        }
        ui = {
        }
        input_meta = self.get_input_meta()
        if self.INPUT_PORT_NAME in input_meta:
            col_from_inport = input_meta[self.INPUT_PORT_NAME]
            enums = [col for col in col_from_inport.keys()]
            json['properties']['target']['enum'] = enums
            return ConfSchema(json=json, ui=ui)
        else:
            return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        split the dataframe to train and tests
        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        input_df = inputs[self.INPUT_PORT_NAME]
        target_col = self.conf['target']
        train_cols = input_df.columns.difference([target_col])
        conf = copy.copy(self.conf)
        del conf['target']
        r = cuml.preprocessing.model_selection.train_test_split(
            input_df[train_cols], input_df[target_col], **conf)
        r[0].index = r[2].index
        r[0][target_col] = r[2]
        r[1].index = r[3].index
        r[1][target_col] = r[3]
        output = {}
        if self.outport_connected(self.OUTPUT_PORT_NAME_TRAIN):
            output.update({self.OUTPUT_PORT_NAME_TRAIN: r[0]})
        if self.outport_connected(self.OUTPUT_PORT_NAME_TEST):
            output.update({self.OUTPUT_PORT_NAME_TEST: r[1]})
        return output
