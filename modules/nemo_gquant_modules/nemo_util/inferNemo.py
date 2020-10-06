from gquant.dataframe_flow import Node
from gquant.dataframe_flow.portsSpecSchema import ConfSchema, PortsSpecSchema
from gquant.dataframe_flow.portsSpecSchema import NodePorts
from gquant.dataframe_flow._port_type_node import _PortTypesMixin

from nemo.core.neural_types import NmTensor
from .trainNemo import NemoTrainNode
import nemo
import copy

__all__ = ["NemoInferNode"]


class NemoInferNode(Node, _PortTypesMixin):
    def init(self):
        _PortTypesMixin.init(self)
        self.OUTPUT_PORT_NAME = 'torch_tensor'
        self.INPUT_PORT_NAME = 'log_dir'

    def ports_setup(self):
        port_type = PortsSpecSchema.port_type
        o_inports = {}
        o_inports[self.INPUT_PORT_NAME] = {port_type: str}
        o_outports = {}
        o_inports['input_tensor'] = {port_type: NmTensor}
        if hasattr(self, 'inputs'):
            for inp in self.inputs:
                if inp['to_port'] == self.INPUT_PORT_NAME:
                    continue
                o_inports[inp['from_node'].uid+'@'+inp['from_port']] = {
                    port_type: NmTensor}
        o_outports = {}
        o_outports[self.OUTPUT_PORT_NAME] = {port_type: list}
        return NodePorts(inports=o_inports, outports=o_outports)

    def columns_setup(self):
        self.required = {}
        output = {}
        output['axes'] = []
        output['element'] = {}
        output['element']['types'] = ['VoidType']
        output['element']['fields'] = 'None'
        output['element']['parameters'] = '{}'
        ports = self.ports_setup()
        inports = ports.inports
        if inports is not None:
            for k in inports.keys():
                if k == self.INPUT_PORT_NAME:
                    continue
                self.required[k] = output
        return {}

    def conf_schema(self):
        json = {
            "title": "NeMo Infer Node",
            "type": "object",
            "description": """Node used to run NeMo neural network inference
             to obtain values for tensors""",
            "properties": {
                "tensors": {
                    "type": "array",
                    "description": """List of NeMo tensors
                    that we want to get values of""",
                    "items": {
                        "type": "string",
                    }
                },
                "checkpoint_dir": {
                    "type": ["string", "null"],
                    "description": """Path to checkpoint directory.
                     Default is None which does not load checkpoints.""",
                    "default": None
                },
                "ckpt_pattern": {
                    "type": "string",
                    "description": """Pattern used to check for checkpoints inside
                checkpoint_dir. Default is '' which matches any checkpoints
                inside checkpoint_dir.""",
                    "default": '',
                },
                "verbose": {
                    "type": "boolean",
                    "description": """Controls printing.""",
                    "default": True
                },
                "cache": {
                    "type": "boolean",
                    "description": """If True, cache all `tensors` and intermediate
                     tensors so that future calls that have use_cache set will
                      avoid computation.""",
                    "default": False
                },
                "use_cache": {
                    "type": "boolean",
                    "description": """Values from `tensors` will be always re-computed.
                It will re-use intermediate tensors from the DAG leading to
                `tensors`. If you want something to be re-computed, put it into
                `tensors` list.""",
                    "default": False
                },
                "offload_to_cpu": {
                    "type": "boolean",
                    "description": """If True, all evaluated tensors are moved to
                cpu memory after each inference batch.""",
                    "default": True
                }
             }
        }
        ui = {
            "checkpoint_dir": {"ui:widget": "PathSelector"},
        }
        enum = []
        enumNames = []
        count = 1
        if hasattr(self, 'inputs'):
            for i in self.inputs:
                if i['to_port'] == self.INPUT_PORT_NAME:
                    continue
                enum.append(i['from_node'].uid+'@'+i['from_port'])
                enumNames.append(i['from_node'].uid+'.'+i['from_port'])
                count += 1
        json['properties']["tensors"]['items']['enum'] = enum
        json['properties']["tensors"]['items']['enumNames'] = enumNames
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        nf = nemo.core.NeuralModuleFactory.get_default_factory()
        input_columns = self.get_input_columns()
        conf = copy.copy(self.conf)
        if self.INPUT_PORT_NAME in input_columns:
            conf['checkpoint_dir'] = inputs[self.INPUT_PORT_NAME]
        conf['tensors'] = [inputs[i] for i in conf['tensors']]
        result = nf.infer(**conf)
        return {self.OUTPUT_PORT_NAME: result}
