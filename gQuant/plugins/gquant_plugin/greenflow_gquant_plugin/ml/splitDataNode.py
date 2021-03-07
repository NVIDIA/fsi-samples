from ..simpleNodeMixin import SimpleNodeMixin
from greenflow.dataframe_flow.portsSpecSchema import (ConfSchema,
                                                      PortsSpecSchema)
from greenflow.dataframe_flow import Node
import cuml
import copy


__all__ = ['DataSplittingNode']


class DataSplittingNode(SimpleNodeMixin, Node):

    def init(self):
        SimpleNodeMixin.init(self)
        self.delayed_process = True
        port_type = PortsSpecSchema.port_type
        self.INPUT_PORT_NAME = 'in'
        self.OUTPUT_PORT_NAME_TRAIN = 'train'
        self.OUTPUT_PORT_NAME_TEST = 'test'
        self.port_inports = {
            self.INPUT_PORT_NAME: {
                port_type: [
                    "pandas.DataFrame", "cudf.DataFrame",
                    "dask_cudf.DataFrame", "dask.dataframe.DataFrame"
                ]
            },
        }
        self.port_outports = {
            self.OUTPUT_PORT_NAME_TRAIN: {
                port_type: "${port:in}"
            },
            self.OUTPUT_PORT_NAME_TEST: {
                port_type: "${port:in}"
            }
        }
        self.meta_inports = {
            self.INPUT_PORT_NAME: {}
        }
        self.meta_outports = {
            self.OUTPUT_PORT_NAME_TRAIN: {
                self.META_OP: self.META_OP_DELETION,
                self.META_REF_INPUT: self.INPUT_PORT_NAME,
                self.META_DATA: {}
            },
            self.OUTPUT_PORT_NAME_TEST: {
                self.META_OP: self.META_OP_DELETION,
                self.META_REF_INPUT: self.INPUT_PORT_NAME,
                self.META_DATA: {}
            }
        }
        if 'target' in self.conf:
            target_col = self.conf['target']
            self.meta_inports = {
                self.INPUT_PORT_NAME: {
                    target_col: None
                }
            }
            self.meta_outports[self.OUTPUT_PORT_NAME_TEST][self.META_ORDER] = {
                target_col: -1
            }
            self.meta_outports[self.OUTPUT_PORT_NAME_TRAIN][
                self.META_ORDER] = {
                    target_col: -1,
                }

    def ports_setup(self):
        return SimpleNodeMixin.ports_setup(self)

    def meta_setup(self):
        return SimpleNodeMixin.meta_setup(self)
        # outputs = meta.outports[self.OUTPUT_PORT_NAME_TRAIN]
        # myList = list(outputs.keys())
        # [i[0] for i in sorted(enumerate(myList), key=lambda x:x[1])]

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
        train_cols = list(input_df.columns)
        if target_col in train_cols:
            train_cols.remove(target_col)
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
