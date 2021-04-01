from jsonpath_ng import parse
import uuid
import cudf
import pandas
from copy import deepcopy
import ray
from ray import tune

from greenflow.plugin_nodes.util.contextCompositeNode import \
    ContextCompositeNode
from greenflow.plugin_nodes.util.compositeNode import (group_ports, _get_node,
                                                       _get_port)
from greenflow.dataframe_flow.portsSpecSchema import (
    ConfSchema, PortsSpecSchema, NodePorts)
from greenflow.dataframe_flow.metaSpec import MetaData
from greenflow.dataframe_flow import TaskGraph
from greenflow.dataframe_flow import Node
from greenflow.dataframe_flow.util import get_file_path
from greenflow.dataframe_flow.taskSpecSchema import TaskSpecSchema
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin

__all__ = ["GridRandomSearchNode"]


_CONF_JSON = {
    "description": """
    Use Tune to specify a grid search
    or random search for a context composite node.
    """,
    "definitions": {
        "number": {
            "type": "object",
            "oneOf": [
                {
                    "title": "randn",
                    "description": """Wraps
                     tune.sample_from around
                     np.random.randn.
                     tune.randn(10)
                      is equivalent to
                      np.random.randn(10)""",
                    "properties": {
                        "function": {
                            "type": "string",
                            "enum": [
                                "randn"
                            ],
                            "default": "randn"
                        },
                        "args": {
                            "type": "array",
                            "items": [
                                {
                                    "description": "number of samples",
                                    "type": "number",
                                    "default": 1.0
                                }
                            ]
                        }
                    }
                },
                {
                    "title": "uniform",
                    "description": """Wraps tune.sample_from around
                     np.random.uniform""",
                    "properties": {
                        "function": {
                            "type": "string",
                            "enum": [
                                "uniform"
                            ],
                            "default": "uniform"
                        },
                        "args": {
                            "type": "array",
                            "items": [
                                {
                                    "type": "number",
                                    "description": "Lower boundary",
                                    "default": 0.0
                                },
                                {
                                    "type": "number",
                                    "description": "Upper boundary",
                                    "default": 10.0
                                }
                            ]
                        }
                    }
                },
                {
                    "title": "loguniform",
                    "description": """Sugar for sampling
                    in different orders of magnitude.,
                    parameters, min_bound – Lower
                    boundary of the output interval,
                    max_bound (float) – Upper boundary
                    of the output interval (1e-2),
                    base – Base of the log.""",
                    "properties": {
                        "function": {
                            "type": "string",
                            "enum": [
                                "loguniform"
                            ],
                            "default": "loguniform"
                        },
                        "args": {
                            "type": "array",
                            "items": [
                                {
                                    "type": "number",
                                    "description": "Lower boundary",
                                    "default": 0.0001
                                },
                                {
                                    "type": "number",
                                    "description": "Upper boundary",
                                    "default": 0.01
                                },
                                {
                                    "type": "number",
                                    "description": "Log base",
                                    "default": 10
                                }
                            ]
                        }
                    }
                },
                {
                    "title": "choice",
                    "description": """Wraps tune.sample_from
                    around random.choice.""",
                    "properties": {
                        "function": {
                            "type": "string",
                            "enum": [
                                "choice"
                            ],
                            "default": "choice"
                        },
                        "args": {
                            "type": "array",
                            "items": {
                                "type": "number"
                            }
                        }
                    }
                },
                {
                    "title": "grid_search",
                    "description": """Convenience method for
                    specifying grid search over a value.""",
                    "properties": {
                        "function": {
                            "type": "string",
                            "enum": [
                                "grid_search"
                            ],
                            "default": "grid_search"
                        },
                        "args": {
                            "type": "array",
                            "items": {
                                "type": "number"
                            }
                        }
                    }
                }
            ]
        },
        "string": {
            "type": "object",
            "oneOf": [
                {
                    "title": "choice",
                    "description": """Wraps tune.sample_from
                    around random.choice.""",
                    "properties": {
                        "function": {
                            "type": "string",
                            "enum": [
                                "choice"
                            ],
                            "default": "choice"
                        },
                        "args": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        }
                    }
                },
                {
                    "title": "grid_search",
                    "description": """Convenience method for
                    specifying grid search over a value.""",
                    "properties": {
                        "function": {
                            "type": "string",
                            "enum": [
                                "grid_search"
                            ],
                            "default": "grid_search"
                        },
                        "args": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        }
                    }
                }
            ]
        }
    },
    "type": "object",
    "properties": {
        "parameters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string"
                    }
                },
                "dependencies": {
                    "name": {
                        "oneOf": []
                    }
                }
            }
        },
        "metrics": {
            "type": "array",
            "description": """the metrics that is going to be
             recorded""",
            "items": {
                "type": "string"
            },
            "default": []
        },
        "best": {
            "description": """the metric that is used for
             best configuration""",
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string"
                },
                "mode": {
                    "type": "string",
                    "enum": [
                        "min",
                        "max"
                    ],
                    "default": "max"
                }
            }
        },
        "tune": {
            "type": "object",
            "properties": {
                "local_dir": {
                    "type": "string",
                    "description": """
                     Local dir to save training results to.
                    """,
                    "default": "./ray"
                },
                "name": {
                    "type": "string",
                    "description": "Name of experiment",
                    "default": "exp"
                },
                "num_samples": {
                    "type": "number",
                    "description": """
                     Number of times to sample from
                     the hyperparameter space.
                     If grid_search is provided
                     as an argument, the grid will be
                     repeated num_samples of times.
                    """,
                    "default": 1
                },
                "resources_per_trial": {
                    "type": "object",
                    "description": """
                    Machine resources to allocate per trial,
                     e.g. {"cpu": 64, "gpu": 8}. Note that
                     GPUs will not be assigned unless you
                     specify them here.""",
                    "properties": {
                        "cpu": {
                            "type": "number",
                            "default": 1
                        },
                        "gpu": {
                            "type": "number",
                            "default": 1
                        }
                    }
                }
            }
        }
    }
}


class SeriliazableNode(object):

    def __init__(self, uid, meta_data, ports_data):
        self.meta_data = meta_data
        self.ports_data = ports_data
        self.uid = uid

    def ports_setup(self):
        return self.ports_data

    def meta_setup(self):
        return self.meta_data


def get_clean_inputs(conf, task_graph, inputs):
    update_inputs = []
    if 'input' in conf:
        # group input ports by node id
        inp_groups = group_ports(conf['input'])
        for inp in inp_groups.keys():
            if inp in task_graph:
                inputNode = task_graph[inp]
                replaced_ports = set(inp_groups[inp])
                for oldInput in inputNode.inputs:
                    if oldInput['to_port'] in replaced_ports:
                        # we want to disconnect this old one and
                        # connect to external node
                        if True:
                            for externalInput in inputs:
                                if (_get_node(
                                    externalInput[
                                        'to_port']) == inputNode.uid
                                        and _get_port(
                                            externalInput[
                                                'to_port']) == oldInput[
                                                    'to_port']):
                                    newInput = {}
                                    newInput['to_port'] = externalInput[
                                        'to_port']
                                    newInput['from_port'] = externalInput[
                                        'from_port']
                                    e_node = externalInput['from_node']
                                    newInput['from_node'] = SeriliazableNode(
                                        e_node.uid,
                                        e_node.meta_setup(),
                                        e_node.ports_setup())
                                    update_inputs.append(newInput)
    return update_inputs


def update_conf_for_search(conf, replaceObj, task_graph, config):
    # find the other replacment conf
    if task_graph:
        for task in task_graph:
            key = task.get('id')
            newid = key
            conf_obj = task.get('conf')
            if newid in replaceObj:
                replaceObj[newid].update({'conf': conf_obj})
            else:
                replaceObj[newid] = {}
                replaceObj[newid].update({'conf': conf_obj})
    # replace the numbers from the context
    if 'context' in conf:
        for key in conf['context'].keys():
            val = conf['context'][key]['value']
            if key in config:
                val = config[key]
            for map_obj in conf['context'][key]['map']:
                xpath = map_obj['xpath']
                expr = parse(xpath)
                expr.update(replaceObj, val)


def get_search_fun(data_store, conf, inputs):

    def search_fun(config, checkpoint_dir=None):
        myinputs = {}
        for key in data_store.keys():
            v = ray.get(data_store[key])
            if isinstance(v, pandas.DataFrame):
                myinputs[key] = cudf.from_pandas(v)
            else:
                myinputs[key] = v
        task_graph = TaskGraph.load_taskgraph(
            get_file_path(conf['taskgraph']))
        task_graph._build()

        outputLists = []
        replaceObj = {}
        input_feeders = []

        def inputNode_fun(inputNode, in_ports):
            inports = inputNode.ports_setup().inports

            class InputFeed(TemplateNodeMixin, Node):

                def meta_setup(self):
                    output = {}
                    for inp in inputNode.inputs:
                        # output[inp['to_port']] = inp[
                        #     'from_node'].meta_setup().outports[
                        #         inp['from_port']]
                        output[inp['to_port']] = inp[
                            'from_node'].meta_setup().outports[
                                inp['from_port']]
                    # it will be something like { input_port: columns }
                    return MetaData(inports={}, outports=output)

                def ports_setup(self):
                    # it will be something like { input_port: types }
                    return NodePorts(inports={}, outports=inports)

                def conf_schema(self):
                    return ConfSchema()

                def update(self):
                    TemplateNodeMixin.update(self)

                def process(self, empty):
                    output = {}
                    for key in inports.keys():
                        if (inputNode.uid+'@'+key
                                in myinputs):
                            output[key] = myinputs[
                                inputNode.uid+'@'+key]
                    return output

            uni_id = str(uuid.uuid1())
            obj = {
                TaskSpecSchema.task_id: uni_id,
                TaskSpecSchema.conf: {},
                TaskSpecSchema.node_type: InputFeed,
                TaskSpecSchema.inputs: []
            }
            input_feeders.append(obj)
            newInputs = {}
            for key in inports.keys():
                if inputNode.uid+'@'+key in myinputs:
                    newInputs[key] = uni_id+'.'+key
            for inp in inputNode.inputs:
                if inp['to_port'] not in in_ports:
                    # need to keep the old connections
                    newInputs[inp['to_port']] = (
                        inp['from_node'].uid + '.' + inp['from_port'])
            replaceObj.update({inputNode.uid: {
                TaskSpecSchema.inputs: newInputs}
            })

        def outNode_fun(outNode, out_ports):
            pass
        outputLists = conf['metrics']

        make_sub_graph_connection(conf, inputs, task_graph, inputNode_fun,
                                  outNode_fun)

        task_graph.extend(input_feeders)
        update_conf_for_search(conf, replaceObj, task_graph, config)
        result = task_graph.run(outputLists, replace=replaceObj)
        metric_report = {item: result[item] for item in outputLists}
        tune.report(**metric_report)
    return search_fun


def make_sub_graph_connection(conf, inputs, task_graph, inputNode_fun,
                              outNode_fun):
    """
    connects the current composite node's inputs and outputs to
    the subgraph-task_graph's inputs and outputs.
    inputNode_fun has subgraph inputNode and all the input ports
    as argument, it processes the inputNode logics
    outputNode_fun has subgraph outputNode and all the outpout ports
    as argument, it processes the outNode logics
    """
    all_inputs = []
    all_outputs = []
    extra_updated = set()
    extra_roots = []
    if 'input' in conf:
        # group input ports by node id
        inp_groups = group_ports(conf['input'])
        for inp in inp_groups.keys():
            if inp in task_graph:
                inputNode = task_graph[inp]
                update_inputs = []
                replaced_ports = set(inp_groups[inp])
                for oldInput in inputNode.inputs:
                    if oldInput['to_port'] in replaced_ports:
                        # we want to disconnect this old one and
                        # connect to external node
                        if True:
                            for externalInput in inputs:
                                if (_get_node(
                                    externalInput[
                                        'to_port']) == inputNode.uid
                                        and _get_port(
                                            externalInput[
                                                'to_port']) == oldInput[
                                                    'to_port']):
                                    newInput = {}
                                    newInput['to_port'] = _get_port(
                                        externalInput['to_port'])
                                    newInput['from_port'] = externalInput[
                                        'from_port']
                                    newInput['from_node'] = externalInput[
                                        'from_node']
                                    update_inputs.append(newInput)
                    else:
                        update_inputs.append(oldInput)
                inputNode.inputs = update_inputs

                # add all the `updated` parents to the set
                for i in inputNode.inputs:
                    if isinstance(i['from_node'], SeriliazableNode):
                        extra_updated.add(i['from_node'])
                # if all the parents are updated, this is
                # a new root node
                if all([
                        isinstance(i['from_node'], SeriliazableNode)
                        for i in inputNode.inputs
                ]):
                    extra_roots.append(inputNode)

                all_inputs.append((inputNode, inp))

    if 'output' in conf:
        oup_groups = group_ports(conf['output'])
        for oup in oup_groups.keys():
            if oup in task_graph:
                outNode = task_graph[oup]
                # we do not disconnect anything here, as we take extra
                # outputs for composite node.
                # Node, we rely on the fact that taskgraph.run method
                # will remove the output collector from taskgraph if
                # the outputlist is set
                all_outputs.append((outNode, oup))
                # outNode_fun(outNode, oup_groups[oup])

    # update all the nodes and cache it
    task_graph.breadth_first_update(extra_roots=extra_roots,
                                    extra_updated=extra_updated)
    for innode in all_inputs:
        inputNode_fun(innode[0], inp_groups[innode[1]])
    for outnode in all_outputs:
        # inputNode_fun(innode[0], inp_groups[innode[1]])
        outNode_fun(outnode[0], oup_groups[outnode[1]])


class GridRandomSearchNode(ContextCompositeNode):

    def init(self):
        ContextCompositeNode.init(self)

    def ports_setup(self):
        return ContextCompositeNode.ports_setup(self)

    def conf_schema(self):
        task_graph = self.task_graph
        # replacementObj = self.replacementObj
        # # cache_key, task_graph, replacementObj = self._compute_has
        # cache_key, task_graph, replacementObj = self._compute_hash_key()
        # if cache_key in CACHE_SCHEMA:
        #     return CACHE_SCHEMA[cache_key]
        # get's the input when it gets the conf
        input_meta = self.get_input_meta()
        json = {}
        if self.INPUT_CONFIG in input_meta:
            conf = input_meta[self.INPUT_CONFIG]
            if 'context' in conf:
                json = deepcopy(_CONF_JSON)
                metrics = []
                # task_graph.build(replace=replacementObj)
                for t in task_graph:
                    node_id = t.get('id')
                    if node_id != '':
                        node = task_graph[node_id]
                        all_ports = node.ports_setup()
                        for port in all_ports.outports.keys():
                            types = all_ports.outports[port][
                                PortsSpecSchema.port_type]
                            if types == float:
                                metrics.append(node_id+'.'+port)
                            elif (isinstance(types, list)
                                  and types[0] == float):
                                metrics.append(node_id+'.'+port)
                context = conf['context']
                json['properties']['parameters'][
                    'items']['properties']['name']['enum'] = list(
                        context.keys())
                json['properties']['metrics'][
                    'items']['enum'] = metrics
                if 'metrics' in self.conf:
                    json['properties']['best'][
                        'properties']['metric']['enum'] = self.conf['metrics']
                options = json['properties']['parameters'][
                    'items']['dependencies']['name']['oneOf']
                for var in context.keys():
                    if (context[var]['type'] == 'number' or
                            context[var]['type'] == 'string'):
                        obj = {
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "enum": [var]
                                },
                                "search": {
                                    "$ref": "#/definitions/{}".format(
                                        context[var]['type'])
                                }
                            }
                        }
                        options.append(obj)
        ui = {
            "tune": {
                "local_dir": {"ui:widget": "PathSelector"}
            }
        }
        out_schema = ConfSchema(json=json, ui=ui)
        # CACHE_SCHEMA[cache_key] = out_schema
        return out_schema

    def meta_setup(self):
        from ray.tune import Analysis
        out_meta = ContextCompositeNode.meta_setup(self)
        if 'tune' in self.conf:
            if 'local_dir' in self.conf['tune']:
                path = self.conf['tune']['local_dir']
                if 'name' in self.conf['tune']:
                    exp = self.conf['tune']['name']
                    try:
                        analysis = Analysis(path+'/'+exp)
                        if 'best' in self.conf:
                            best = analysis.get_best_config(
                                **self.conf['best'])
                            for key in best.keys():
                                self.conf['context'][key]['value'] = best[key]
                            print('get best', best)
                            out_meta.outports[self.OUTPUT_CONFIG] = self.conf
                    except Exception:
                        pass
        return out_meta

    def process(self, inputs):
        if self.INPUT_CONFIG in inputs:
            self.conf.update(inputs[self.INPUT_CONFIG].data)
        output = {}
        if self.outport_connected(self.OUTPUT_CONFIG):
            data_store = {}
            for key in inputs.keys():
                v = inputs[key]
                if isinstance(v, cudf.DataFrame):
                    # it is a work around,
                    # the ray.put doesn't support GPU cudf
                    data_store[key] = ray.put(v.to_pandas())
                else:
                    data_store[key] = ray.put(v)
            # here we need to do the hyper parameter search

            config = {}
            for para in self.conf['parameters']:
                fun_name = para['search']['function']
                fun = getattr(tune, fun_name)
                if fun_name == 'grid_search' or fun_name == 'choice':
                    config[para['name']] = fun(para['search']['args'])
                else:
                    config[para['name']] = fun(*para['search']['args'])
            clean_inputs = get_clean_inputs(self.conf, self.task_graph,
                                            self.inputs)
            fun = get_search_fun(data_store, self.conf, clean_inputs)
            analysis = tune.run(fun, **self.conf['tune'], config=config)
            best = analysis.get_best_config(**self.conf['best'])
            for key in best.keys():
                self.conf['context'][key]['value'] = best[key]
            output[self.OUTPUT_CONFIG] = self.conf
        more_output = ContextCompositeNode.process(self, inputs)
        output.update(more_output)
        return output
