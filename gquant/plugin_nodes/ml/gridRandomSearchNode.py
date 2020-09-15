from ..util.contextCompositeNode import ContextCompositeNode
from gquant.dataframe_flow.portsSpecSchema import (ConfSchema,
                                                   PortsSpecSchema, NodePorts)
from gquant.dataframe_flow.cache import cache_schema
from gquant.dataframe_flow import TaskGraph
from gquant.dataframe_flow import Node
from gquant.dataframe_flow.util import get_file_path
from gquant.dataframe_flow.taskSpecSchema import TaskSpecSchema
from jsonpath_ng import parse
import uuid
import cudf
import pandas

__all__ = ["GridRandomSearchNode"]


class GridRandomSearchNode(ContextCompositeNode):

    def conf_schema(self):
        cache_key, task_graph, replacementObj = self._compute_hash_key()
        if cache_key in cache_schema:
            return cache_schema[cache_key]
        # get's the input when it gets the conf
        input_columns = self.get_input_columns()
        json = {}
        if self.INPUT_CONFIG in input_columns:
            conf = input_columns[self.INPUT_CONFIG]
            if 'context' in conf:
                json = {
                    "definitions": {
                        "number": {
                            "type": "object",
                            "oneOf": [
                                  {
                                    "title": 'randn',
                                    "description": """Wraps
                                     tune.sample_from around
                                     np.random.randn.
                                     tune.randn(10)
                                      is equivalent to
                                      np.random.randn(10)""",
                                    "properties": {
                                        "function": {
                                            "type": "string",
                                            "enum": ['randn'],
                                            "default": 'randn'
                                        },
                                        "args": {
                                            "type": "array",
                                            "items": [
                                               {
                                                   "type": "number",
                                                   "default": 1.0
                                               }
                                            ]
                                        }
                                    }
                                  },
                                {
                                      "title": "uniform",
                                      "description": """Wraps tune.sample_from
                                    around np.random.uniform""",
                                      "properties": {
                                          "function": {
                                              "type": "string",
                                              "enum": ['uniform'],
                                              "default": 'uniform'
                                          },
                                          "args": {
                                              "type": "array",
                                              "items": [
                                                  {
                                                      "type": "number",
                                                      "default": 0.0
                                                  },
                                                  {
                                                      "type": "number",
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
                                            "enum": ['loguniform'],
                                            "default": 'loguniform'
                                        },
                                        "args": {
                                            "type": "array",
                                            "items": [
                                                {
                                                    "type": "number",
                                                    "default": 1e-4
                                                },
                                                {
                                                    "type": "number",
                                                    "default": 1e-2
                                                },
                                                {
                                                    "type": "number",
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
                                              "enum": ['choice'],
                                              "default": 'choice'
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
                                              "enum": ['grid_search'],
                                              "default": 'grid_search'
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
                                              "enum": ['choice'],
                                              "default": 'choice'
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
                                              "enum": ['grid_search'],
                                              "default": 'grid_search'
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
                    "description": """
                    Use Tune to specify a grid search
                    or random search for a context composite node.
                    """,
                    "type": "object",
                    "properties": {
                        "parameters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                    }
                                },
                                "dependencies": {
                                    "name": {
                                        "oneOf": [],
                                    }
                                }
                            },
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
                                    "enum": ['min', 'max'],
                                    "default": 'max'
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
                                    "description": """Name of experiment""",
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
                                            "type": 'number',
                                            "default": 1
                                        },
                                        "gpu": {
                                            "type": 'number',
                                            "default": 1
                                        },
                                    }
                                }
                            }
                        }
                    }
                }
                metrics = []
                task_graph.build(replace=replacementObj)
                for t in task_graph:
                    node_id = t.get('id')
                    if node_id != '':
                        node = task_graph[node_id]
                        all_ports = node.ports_setup()
                        for port in all_ports.outports.keys():
                            if all_ports.outports[port][
                                    PortsSpecSchema.port_type] == float:
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
        cache_schema[cache_key] = out_schema
        return out_schema

    def columns_setup(self):
        from ray.tune import Analysis
        out_columns = super().columns_setup()
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
                            out_columns[self.OUTPUT_CONFIG] = self.conf
                    except Exception:
                        pass
        return out_columns

    def process(self, inputs):
        import ray
        from ray import tune
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

            def search_fun(config, checkpoint_dir=None):
                myinputs = {}
                for key in data_store.keys():
                    v = ray.get(data_store[key])
                    if isinstance(v, pandas.DataFrame):
                        myinputs[key] = cudf.from_pandas(v)
                    else:
                        myinputs[key] = v
                task_graph = TaskGraph.load_taskgraph(
                    get_file_path(self.conf['taskgraph']))
                task_graph.build()

                outputLists = []
                replaceObj = {}
                input_feeders = []

                def inputNode_fun(inputNode, in_ports):
                    inports = inputNode.ports_setup().inports

                    class InputFeed(Node):

                        def columns_setup(self):
                            output = {}
                            for inp in inputNode.inputs:
                                output[inp['to_port']] = inp[
                                    'from_node'].columns_setup()[
                                        inp['from_port']]
                            # it will be something like { input_port: columns }
                            return output

                        def ports_setup(self):
                            # it will be something like { input_port: types }
                            return NodePorts(inports={}, outports=inports)

                        def conf_schema(self):
                            return ConfSchema()

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
                outputLists = self.conf['metrics']

                self._make_sub_graph_connection(task_graph,
                                                inputNode_fun, outNode_fun)

                task_graph.extend(input_feeders)
                self.update_conf_for_search(replaceObj, task_graph, config)
                result = task_graph.run(outputLists, replace=replaceObj)
                metric_report = {item: result[item] for item in outputLists}
                tune.report(**metric_report)
            config = {}
            for para in self.conf['parameters']:
                fun_name = para['search']['function']
                fun = getattr(tune, fun_name)
                if fun_name == 'grid_search' or fun_name == 'choice':
                    config[para['name']] = fun(para['search']['args'])
                else:
                    config[para['name']] = fun(*para['search']['args'])
            analysis = tune.run(search_fun, **self.conf['tune'], config=config)
            best = analysis.get_best_config(**self.conf['best'])
            for key in best.keys():
                self.conf['context'][key]['value'] = best[key]
            output[self.OUTPUT_CONFIG] = self.conf
        more_output = super().process(inputs)
        output.update(more_output)
        return output

    def update_conf_for_search(self, replaceObj, task_graph, config):
        # find the other replacment conf
        if task_graph:
            for task in task_graph:
                key = task.get('id')
                newid = key
                conf = task.get('conf')
                if newid in replaceObj:
                    replaceObj[newid].update({'conf': conf})
                else:
                    replaceObj[newid] = {}
                    replaceObj[newid].update({'conf': conf})
        # replace the numbers from the context
        if 'context' in self.conf:
            for key in self.conf['context'].keys():
                val = self.conf['context'][key]['value']
                if key in config:
                    val = config[key]
                for map_obj in self.conf['context'][key]['map']:
                    xpath = map_obj['xpath']
                    expr = parse(xpath)
                    expr.update(replaceObj, val)
