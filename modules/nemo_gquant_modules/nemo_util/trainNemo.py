from gquant.dataframe_flow import Node
from gquant.dataframe_flow.portsSpecSchema import ConfSchema, PortsSpecSchema
from gquant.dataframe_flow.portsSpecSchema import NodePorts,  MetaData

from nemo.core.neural_types import NmTensor
import nemo
import numpy
import copy

__all__ = ["NemoTrainNode"]


class CallBack(object):

    def __init__(self):
        self.counter = 0

    def __call__(self, global_vars):
        import ray
        from ray import tune
        reports = {}
        for key in global_vars.keys():
            value = numpy.array(global_vars[key]).mean()
            print('eval:', key, value)
            reports[key] = value
        if ray.is_initialized():
            reports['training_iteration'] = self.counter
            tune.report(**reports)
            self.counter += 1


class NemoTrainNode(Node):
    def init(self):
        self.OUTPUT_PORT_NAME = 'checkpoint_dir'

    def ports_setup(self):
        dy = PortsSpecSchema.dynamic
        port_type = PortsSpecSchema.port_type
        o_inports = {}
        o_outports = {}
        o_inports['input_tensor'] = {port_type: NmTensor, dy: True}
        # if hasattr(self, 'inputs'):
        #     for inp in self.inputs:
        #         # TODO: Move TaskGrah rewire logic here instead of in
        #         #     chartEngine.tsx ChartEngine._fixNeMoPorts
        #         o_inports[inp['from_node'].uid+'@'+inp['from_port']] = {
        #             port_type: NmTensor}
        o_outports[self.OUTPUT_PORT_NAME] = {port_type: str}
        return NodePorts(inports=o_inports, outports=o_outports)

    def meta_setup(self):
        required = {}
        output = {}
        output['axes'] = []
        output['element'] = {}
        output['element']['types'] = ['VoidType']
        output['element']['fields'] = 'None'
        output['element']['parameters'] = '{}'
        ports = self.calculated_ports_setup()
        inports = ports.inports
        iports_connected = self.get_connected_inports()
        iports_cols = self.get_input_meta()
        dy = PortsSpecSchema.dynamic
        if hasattr(self, 'inputs'):
            for inp in self.inputs:
                oport = inp['to_port']
                iport = inp['from_port']
                out_port_name = inp['from_node'].uid+'@'+iport
                if out_port_name in inports and inports[
                        out_port_name].get(dy, False):
                    if oport in iports_connected and oport in iports_cols:
                        required[out_port_name] = copy.deepcopy(
                            iports_cols[oport])
                else:
                    if oport in iports_connected and oport in iports_cols:
                        required[oport] = copy.deepcopy(iports_cols[oport])
        required['input_tensor'] = copy.deepcopy(output)
        metadata = MetaData(inports=required,
                            outports={self.OUTPUT_PORT_NAME: {}})
        return metadata

    def conf_schema(self):
        json = {
            "title": "NeMo Train Node",
            "type": "object",
            "description": "Node used to train a NeMo neural network",
            "properties": {
                "parameters": {
                    "type": "object",
                    "description": "parameters for train method",
                    "properties": {
                        "tensors_to_optimize": {
                            "type": "array",
                            "items": {
                                "type": "string",
                            }
                        },
                        "batches_per_step": {
                            "type": "number",
                            "default": None
                        },
                        "stop_on_nan_loss": {
                            "type": "boolean",
                            "default": False
                        },
                        "synced_batchnorm": {
                            "type": "boolean",
                            "default": False
                        },
                        "synced_batchnorm_groupsize": {
                            "type": "number",
                            "default": 0
                        },
                        "gradient_predivide": {
                            "type": "boolean",
                            "default": False
                        },
                        "amp_max_loss_scale": {
                            "type": "number",
                            "default": 16777216.0
                        },
                        "reset": {
                            "type": "boolean",
                            "default": False
                        }
                    }
                },
                "check_point": {
                    "type": "object",
                    "description": "parameters for checkpoint method",
                    "properties": {
                        "folder": {
                            "type": "string",
                            "description": """A path where checkpoints are to
                             be stored and loaded from if load_from_folder
                              is None"""
                        },
                        "load_from_folder": {
                            "type": ["string", "null"],
                            "description": """A path where checkpoints can be
                             loaded from""",
                            "default": None
                        },
                        "step_freq": {
                            "type": ["integer", "null"],
                            "description": """How often in terms of steps to
                             save checkpoints. One of step_freq or epoch_freq
                              is required""",
                            "default": None
                        },
                        "epoch_freq": {
                            "type": ["integer", "null"],
                            "description": """How often in terms of epochs to
                             save checkpoints. One of step_freq or epoch_freq
                              is required.""",
                            "default": None
                        },
                        "checkpoints_to_keep": {
                            "type": "integer",
                            "description": """Number of most recent
                            checkpoints to keep. Older checkpoints will be
                            deleted.""",
                            "default": 4
                        },
                        "force_load": {
                            "type": "boolean",
                            "description": """Whether to crash if loading
                            is unsuccessful.""",
                            "default": False
                        }
                    }
                },
                "simple_logger": {
                    "type": "object",
                    "description": """A simple callback that prints tensors
                     to screen. It's default option is to print the training
                      loss every 100 steps. Additional tensors can be printed
                       by adding them to the tensors_to_log argument.""",
                    "properties": {
                        "step_freq": {
                            "type": "integer",
                            "description": """The frequency of printing to
                            screen. Defaults to every 100 steps""",
                            "default": 100
                        },
                        "tensors_to_log": {
                            "type": "array",
                            "description": """A list of NmTensors which will
                             be printed every step_freq steps.""",
                            "items": {
                                "type": "string",
                            }
                        }
                    }
                },
                "eval_callback": {
                    "type": "object",
                    "description": """Used to report the statistics of
                     evaluation dataset""",
                    "properties": {
                        "eval_step": {
                            "type": ["integer", "null"],
                            "description": """The frequency of running eval""",
                            "default": None
                        },
                        "eval_tensors": {
                            "type": "array",
                            "description": """A list of NmTensors which will
                             be evaluated every eval_step steps.""",
                            "items": {
                                "type": "string",
                            }
                        }
                    }
                },
                "warmup_policy": {
                    "type": "object",
                    "description": """Choose a warm up policy""",
                    "properties": {
                        "name": {
                            "type": "string",
                            "enum": ["WarmupPolicy", "WarmupHoldPolicy",
                                     "SquareAnnealing", "SquareRootAnnealing",
                                     "CosineAnnealing", "WarmupAnnealing",
                                     "InverseSquareRootAnnealing",
                                     "PolynomialDecayAnnealing",
                                     "PolynomialHoldDecayAnnealing"]
                        },
                    },
                    "dependencies": {
                        "name": {
                            "oneOf": [
                                {
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "enum": ["WarmupPolicy"]
                                        },
                                        "parameters": {
                                            "type": "object",
                                            "properties": {
                                                "warmup_steps": {
                                                    "type": ["integer",
                                                             "null"],
                                                    "description": """Number of
                                                     training steps in
                                                      warmup stage""",
                                                    "default": None
                                                },
                                                "warmup_ratio": {
                                                    "type": ["number", "null"],
                                                    "description": """Ratio of
                                                     warmup steps to total
                                                     steps""",
                                                    "default": None
                                                },
                                                "total_steps": {
                                                    "type": ["integer",
                                                             "null"],
                                                    "description": """Total number
                                                    of steps while training or
                                                    `None` for infinite
                                                    training""",
                                                    "default": None
                                                }
                                            }
                                        },
                                    }
                                },
                                {
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "enum": ["WarmupHoldPolicy"]
                                        },
                                        "parameters": {
                                            "type": "object",
                                            "properties": {
                                                "warmup_steps": {
                                                    "type": ["integer",
                                                             "null"],
                                                    "description": """Number of
                                                     training steps in
                                                      warmup stage""",
                                                    "default": None
                                                },
                                                "warmup_ratio": {
                                                    "type": ["number",
                                                             "null"],
                                                    "description": """Ratio of
                                                     warmup steps to total
                                                     steps""",
                                                    "default": None
                                                },
                                                "total_steps": {
                                                    "type": ["integer",
                                                             "null"],
                                                    "description": """Total number
                                                    of steps while training or
                                                    `None` for infinite
                                                    training""",
                                                    "default": None
                                                },
                                                "hold_steps": {
                                                    "type": ["integer",
                                                             "null"],
                                                    "description": """Number of
                                                    training steps to hold the
                                                    learning rate after warm
                                                    up""",
                                                    "default": None
                                                },
                                                "hold_ratio": {
                                                    "type": ["integer",
                                                             "null"],
                                                    "description": """Ratio of hold
                                                     steps to total steps""",
                                                    "default": None
                                                },
                                                "min_lr": {
                                                    "type": "number",
                                                    "description": """minimum learing
                                                    rate""",
                                                    "default": 0.0
                                                },
                                            }
                                        },
                                    }
                                },
                                {
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "enum": ["SquareAnnealing"]
                                        },
                                        "parameters": {
                                            "type": "object",
                                            "properties": {
                                                "warmup_steps": {
                                                    "type": ["integer",
                                                             "null"],
                                                    "description": """Number of
                                                     training steps in
                                                      warmup stage""",
                                                    "default": None
                                                },
                                                "warmup_ratio": {
                                                    "type": ["number", "null"],
                                                    "description": """Ratio of
                                                     warmup steps to total
                                                     steps""",
                                                    "default": None
                                                },
                                                "total_steps": {
                                                    "type": ["integer",
                                                             "null"],
                                                    "description": """Total number
                                                    of steps while training or
                                                    `None` for infinite
                                                    training""",
                                                    "default": None
                                                },
                                                "min_lr": {
                                                    "type": "number",
                                                    "description": """minimum learing
                                                    rate""",
                                                    "default": 0.00001
                                                },
                                            }
                                        },
                                    }
                                },
                                {
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "enum": ["SquareRootAnnealing"]
                                        },
                                        "parameters": {
                                            "type": "object",
                                            "properties": {
                                                "warmup_steps": {
                                                    "type": ["integer",
                                                             "null"],
                                                    "description": """Number of
                                                     training steps in
                                                      warmup stage""",
                                                    "default": None
                                                },
                                                "warmup_ratio": {
                                                    "type": ["number", "null"],
                                                    "description": """Ratio of
                                                     warmup steps to total
                                                     steps""",
                                                    "default": None
                                                },
                                                "total_steps": {
                                                    "type": ["integer",
                                                             "null"],
                                                    "description": """Total number
                                                    of steps while training or
                                                    `None` for infinite
                                                    training""",
                                                    "default": None
                                                },
                                                "min_lr": {
                                                    "type": "number",
                                                    "description": """minimum learing
                                                    rate""",
                                                    "default": 0.0
                                                },
                                            }
                                        },
                                    }
                                },
                                {
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "enum": ["CosineAnnealing"]
                                        },
                                        "parameters": {
                                            "type": "object",
                                            "properties": {
                                                "warmup_steps": {
                                                    "type": ["integer",
                                                             "null"],
                                                    "description": """Number of
                                                     training steps in
                                                      warmup stage""",
                                                    "default": None
                                                },
                                                "warmup_ratio": {
                                                    "type": ["number", "null"],
                                                    "description": """Ratio of
                                                     warmup steps to total
                                                     steps""",
                                                    "default": None
                                                },
                                                "total_steps": {
                                                    "type": ["integer",
                                                             "null"],
                                                    "description": """Total number
                                                    of steps while training or
                                                    `None` for infinite
                                                    training""",
                                                    "default": None
                                                },
                                                "min_lr": {
                                                    "type": "number",
                                                    "description": """minimum learing
                                                    rate""",
                                                    "default": 0.0
                                                },
                                            }
                                        },
                                    }
                                },
                                {
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "enum": ["WarmupAnnealing"]
                                        },
                                        "parameters": {
                                            "type": "object",
                                            "properties": {
                                                "warmup_steps": {
                                                    "type": ["integer",
                                                             "null"],
                                                    "description": """Number of
                                                     training steps in
                                                      warmup stage""",
                                                    "default": None
                                                },
                                                "warmup_ratio": {
                                                    "type": ["number", "null"],
                                                    "description": """Ratio of
                                                     warmup steps to total
                                                     steps""",
                                                    "default": None
                                                },
                                                "total_steps": {
                                                    "type": ["integer",
                                                             "null"],
                                                    "description": """Total number
                                                    of steps while training or
                                                    `None` for infinite
                                                    training""",
                                                    "default": None
                                                },
                                            }
                                        },
                                    }
                                },
                                {
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "enum": [
                                                "InverseSquareRootAnnealing"]
                                        },
                                        "parameters": {
                                            "type": "object",
                                            "properties": {
                                                "warmup_steps": {
                                                    "type": ["integer",
                                                             "null"],
                                                    "description": """Number of
                                                     training steps in
                                                      warmup stage""",
                                                    "default": None
                                                },
                                                "warmup_ratio": {
                                                    "type": ["number", "null"],
                                                    "description": """Ratio of
                                                     warmup steps to total
                                                     steps""",
                                                    "default": None
                                                },
                                                "total_steps": {
                                                    "type": ["integer",
                                                             "null"],
                                                    "description": """Total number
                                                    of steps while training or
                                                    `None` for infinite
                                                    training""",
                                                    "default": None
                                                },
                                            }
                                        },
                                    }
                                },
                                {
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "enum": [
                                                "PolynomialDecayAnnealing"]
                                        },
                                        "parameters": {
                                            "type": "object",
                                            "properties": {
                                                "warmup_steps": {
                                                    "type": ["integer",
                                                             "null"],
                                                    "description": """Number of
                                                     training steps in
                                                      warmup stage""",
                                                    "default": None
                                                },
                                                "warmup_ratio": {
                                                    "type": ["number", "null"],
                                                    "description": """Ratio of
                                                     warmup steps to total
                                                     steps""",
                                                    "default": None
                                                },
                                                "total_steps": {
                                                    "type": ["integer",
                                                             "null"],
                                                    "description": """Total number
                                                    of steps while training or
                                                    `None` for infinite
                                                    training""",
                                                    "default": None
                                                },
                                                "min_lr": {
                                                    "type": "number",
                                                    "description": """minimum learing
                                                    rate""",
                                                    "default": 0.0
                                                },
                                                "power": {
                                                    "type": "number",
                                                    "default": 1.0
                                                },
                                                "cycle": {
                                                    "type": "boolean",
                                                    "default": False
                                                },
                                            }
                                        },
                                    }
                                },
                                {
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "enum": [
                                                "PolynomialHoldDecayAnnealing"]
                                        },
                                        "parameters": {
                                            "type": "object",
                                            "properties": {
                                                "warmup_steps": {
                                                    "type": ["integer",
                                                             "null"],
                                                    "description": """Number of
                                                     training steps in
                                                      warmup stage""",
                                                    "default": None
                                                },
                                                "warmup_ratio": {
                                                    "type": ["number", "null"],
                                                    "description": """Ratio of
                                                     warmup steps to total
                                                     steps""",
                                                    "default": None
                                                },
                                                "total_steps": {
                                                    "type": ["integer",
                                                             "null"],
                                                    "description": """Total number
                                                    of steps while training or
                                                    `None` for infinite
                                                    training""",
                                                    "default": None
                                                },
                                                "hold_steps": {
                                                    "type": ["integer",
                                                             "null"],
                                                    "description": """Number of
                                                    training steps to hold the
                                                    learning rate after warm
                                                    up""",
                                                    "default": None
                                                },
                                                "hold_ratio": {
                                                    "type": ["integer",
                                                             "null"],
                                                    "description": """Ratio of hold
                                                     steps to total steps""",
                                                    "default": None
                                                },
                                                "min_lr": {
                                                    "type": "number",
                                                    "description": """minimum learing
                                                    rate""",
                                                    "default": 0.0
                                                },
                                                "power": {
                                                    "type": "number",
                                                    "default": 1.0
                                                },
                                                "cycle": {
                                                    "type": "boolean",
                                                    "default": False
                                                },
                                            }
                                        },
                                    }
                                }
                            ]
                        }
                    }
                },
                "optimizer": {
                    "type": "object",
                    "description": """The optimization algorithm""",
                    "properties": {
                        "name": {
                            "type": "string",
                            "enum": ["sgd", "adam", "fused_adam", "adam_w",
                                     "novograd", "fused_novograd",
                                     "fused_lamb"]
                        },
                    },
                    "dependencies": {
                        "name": {
                            "oneOf": [
                                {
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "enum": ["sgd"]
                                        },
                                        "parameters": {
                                            "type": "object",
                                            "properties": {
                                                "num_epochs": {
                                                    "type": "integer",
                                                    "default": 10
                                                },
                                                "lr": {
                                                    "type": "number",
                                                    "default": 0.001
                                                },
                                                "momentum": {
                                                    "type": "number",
                                                    "default": 0.9
                                                },
                                                "weight_decay": {
                                                    "type": "number",
                                                    "default": 0.0
                                                },
                                            }
                                        },
                                    }
                                },
                                {
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "enum": ["adam"]
                                        },
                                        "parameters": {
                                            "type": "object",
                                            "properties": {
                                                "num_epochs": {
                                                    "type": "integer",
                                                    "default": 10
                                                },
                                                "lr": {
                                                    "type": "number",
                                                    "default": 0.001
                                                },
                                                "betas": {
                                                    "type": "array",
                                                    "items": [
                                                        {
                                                            "type": "number",
                                                            "default": 0.9
                                                        },
                                                        {
                                                            "type": "number",
                                                            "default": 0.999
                                                        }
                                                    ]
                                                },
                                                "eps": {
                                                    "type": "number",
                                                    "default": 0.000000001
                                                },
                                                "weight_decay": {
                                                    "type": "number",
                                                    "default": 0.0
                                                },
                                                "amsgrad": {
                                                    "type": "boolean",
                                                    "default": False
                                                },
                                            }
                                        },
                                    }
                                },
                                {
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "enum": ["fused_adam"]
                                        },
                                        "parameters": {
                                            "type": "object",
                                            "properties": {
                                                "num_epochs": {
                                                    "type": "integer",
                                                    "default": 10
                                                },
                                                "lr": {
                                                    "type": "number",
                                                    "default": 0.001
                                                },
                                                "betas": {
                                                    "type": "array",
                                                    "items": [
                                                        {
                                                            "type": "number",
                                                            "default": 0.9
                                                        },
                                                        {
                                                            "type": "number",
                                                            "default": 0.999
                                                        }
                                                    ]
                                                },
                                                "eps": {
                                                    "type": "number",
                                                    "default": 0.00000001,
                                                },
                                                "weight_decay": {
                                                    "type": "number",
                                                    "default": 0.0
                                                }
                                            }
                                        },
                                    }
                                },
                                {
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "enum": ["adam_w"]
                                        },
                                        "parameters": {
                                            "type": "object",
                                            "properties": {
                                                "num_epochs": {
                                                    "type": "integer",
                                                    "default": 10
                                                },
                                                "lr": {
                                                    "type": "number",
                                                    "default": 0.001
                                                },
                                                "betas": {
                                                    "type": "array",
                                                    "items": [
                                                        {
                                                            "type": "number",
                                                            "default": 0.9
                                                        },
                                                        {
                                                            "type": "number",
                                                            "default": 0.999
                                                        }
                                                    ]
                                                },
                                                "eps": {
                                                    "type": "number",
                                                    "default": 0.00000001,
                                                },
                                                "weight_decay": {
                                                    "type": "number",
                                                    "default": 0.0
                                                },
                                                "amsgrad": {
                                                    "type": "boolean",
                                                    "default": False
                                                },
                                            }
                                        },
                                    }
                                },
                                {
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "enum": ["novograd"]
                                        },
                                        "parameters": {
                                            "type": "object",
                                            "properties": {
                                                "num_epochs": {
                                                    "type": "integer",
                                                    "default": 10
                                                },
                                                "lr": {
                                                    "type": "number",
                                                    "default": 0.001
                                                },
                                                "betas": {
                                                    "type": "array",
                                                    "items": [
                                                        {
                                                            "type": "number",
                                                            "default": 0.9
                                                        },
                                                        {
                                                            "type": "number",
                                                            "default": 0.999
                                                        }
                                                    ]
                                                },
                                                "luc": {
                                                    "type": "boolean",
                                                    "default": False
                                                },
                                                "luc_eta": {
                                                    "type": "number",
                                                    "default": 0.001
                                                },
                                                "weight_decay": {
                                                    "type": "number",
                                                    "default": 0.0
                                                },
                                            }
                                        },
                                    }
                                },
                                {
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "enum": ["fused_novograd"]
                                        },
                                        "parameters": {
                                            "type": "object",
                                            "properties": {
                                                "num_epochs": {
                                                    "type": "integer",
                                                    "default": 10
                                                },
                                                "lr": {
                                                    "type": "number",
                                                    "default": 0.001
                                                },
                                                "betas": {
                                                    "type": "array",
                                                    "items": [
                                                        {
                                                            "type": "number",
                                                            "default": 0.9
                                                        },
                                                        {
                                                            "type": "number",
                                                            "default": 0.999
                                                        }
                                                    ]
                                                },
                                                "reg_inside_moment": {
                                                    "type": "boolean",
                                                    "default": True
                                                },
                                                "grad_averaging": {
                                                    "type": "boolean",
                                                    "default": False
                                                },
                                                "weight_decay": {
                                                    "type": "number",
                                                    "default": 0.0
                                                },
                                            }
                                        },
                                    }
                                },
                                {
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "enum": ["fused_lamb"]
                                        },
                                        "parameters": {
                                            "type": "object",
                                            "properties": {
                                                "num_epochs": {
                                                    "type": "integer",
                                                    "default": 10
                                                },
                                                "lr": {
                                                    "type": "number",
                                                    "default": 0.001
                                                },
                                            }
                                        },
                                    }
                                }
                            ]
                        }
                    }
                }
            }
        }
        ui = {
            "check_point": {
                "folder": {"ui:widget": "PathSelector"},
                "load_from_folder": {"ui:widget": "PathSelector"},
            },
            "warmup_policy": {
                "parameters": {
                    "warmup_steps": {"ui:widget": "updown"},
                    "total_steps": {"ui:widget": "updown"},
                    "warmup_ratio": {"ui:widget": "updown"},
                    "hold_steps": {"ui:widget": "updown"},
                    "hold_ratio": {"ui:widget": "updown"}
                },
            },
            "eval_callback": {
                "eval_step": {"ui:widget": "updown"},
            }
        }
        enum = []
        enumNames = []
        count = 1
        if hasattr(self, 'inputs'):
            for i in self.inputs:
                enum.append(i['from_node'].uid+'@'+i['from_port'])
                enumNames.append(i['from_node'].uid+'.'+i['from_port'])
                count += 1
        json['properties']['parameters'][
            'properties']["tensors_to_optimize"][
                'items']['enum'] = enum
        json['properties']['parameters'][
            'properties']["tensors_to_optimize"][
                'items']['enumNames'] = enumNames
        json['properties']['simple_logger'][
            'properties']["tensors_to_log"][
                'items']['enum'] = enum
        json['properties']['simple_logger'][
            'properties']["tensors_to_log"][
                'items']['enumNames'] = enumNames
        json['properties']['eval_callback'][
            'properties']["eval_tensors"][
                'items']['enum'] = enum
        json['properties']['eval_callback'][
            'properties']["eval_tensors"][
                'items']['enumNames'] = enumNames
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        nf = nemo.core.NeuralModuleFactory.get_default_factory()
        log_conf = copy.copy(self.conf["simple_logger"])
        log_conf['tensors_to_log'] = [
            inputs[i] for i in log_conf['tensors_to_log']]
        log_callback = nemo.core.SimpleLogger(**log_conf)
        check_callback = nemo.core.CheckpointCallback(
            **self.conf['check_point'])
        all_args = copy.copy(self.conf['parameters'])
        all_args['tensors_to_optimize'] = [
            inputs[i] for i in all_args['tensors_to_optimize']]

        # eval callbacks
        def eval_iter_callback(tensors, global_vars):
            for e_name in eval_names:
                if e_name not in global_vars:
                    global_vars[e_name] = []

            for e_name in eval_names:
                for kv, v in tensors.items():
                    if kv.startswith(e_name):
                        global_vars[e_name].append(v[0].cpu().numpy().mean())

        all_args['callbacks'] = [check_callback, log_callback]
        if ('eval_callback' in self.conf and
            'eval_tensors' in self.conf['eval_callback'] and
                len(self.conf['eval_callback']['eval_tensors']) > 0):
            eval_conf = copy.copy(self.conf['eval_callback'])
            eval_conf['eval_tensors'] = [
                inputs[i] for i in eval_conf['eval_tensors']]
            eval_names = [i.name for i in eval_conf['eval_tensors']]
            eval_conf['user_iter_callback'] = eval_iter_callback
            eval_conf['user_epochs_done_callback'] = CallBack()
            eval_callback = nemo.core.EvaluatorCallback(**eval_conf)
            all_args['callbacks'].append(eval_callback)
        all_args['optimizer'] = self.conf['optimizer']['name']
        all_args['optimization_params'] = self.conf['optimizer']['parameters']

        if 'warmup_policy' in self.conf and 'name' in self.conf[
                "warmup_policy"]:
            policy_name = self.conf["warmup_policy"]['name']
            from nemo.utils import lr_policies
            policy_class = getattr(lr_policies, policy_name)
            lr_policy = policy_class(
                **self.conf["warmup_policy"]['parameters'])
            all_args['lr_policy'] = lr_policy
        nf.train(**all_args)
        log_directory = ''
        if (('step_freq' in self.conf['check_point']
             and self.conf['check_point']['step_freq'] is not None)
            or ('epoch_freq' in self.conf['check_point'] and
                self.conf['check_point']['epoch_freq'] is not None)):
            log_directory = self.conf['check_point']['folder']
        return {
            self.OUTPUT_PORT_NAME: log_directory,
        }
