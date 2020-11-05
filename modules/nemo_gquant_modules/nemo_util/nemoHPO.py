from gquant.dataframe_flow.task import load_modules
import os
load_modules(os.getenv('MODULEPATH')+'/rapids_modules/')
from rapids_modules import GridRandomSearchNode
from gquant.plugin_nodes.util.contextCompositeNode import ContextCompositeNode
from gquant.dataframe_flow.portsSpecSchema import (ConfSchema,
                                                   NodePorts)
from gquant.dataframe_flow import TaskGraph
from gquant.dataframe_flow import Node
from gquant.dataframe_flow.util import get_file_path
from gquant.dataframe_flow.cache import cache_schema
from gquant.dataframe_flow.taskSpecSchema import TaskSpecSchema
import cudf
import uuid
import pandas
import copy

__all__ = ["NemoHyperTuneNode"]


_SCHED_CONF = {
    "type": "object",
    "description": """distributed implementations of early
     stopping algorithms such as Median Stopping Rule,
      HyperBand, and ASHA.""",
    "properties": {
        "name": {
            "type": "string",
            "enum": ["AsyncHyperBandScheduler",
                     "HyperBandScheduler",
                     "MedianStoppingRule"]
        },
    },
    "dependencies": {
        "name": {
            "oneOf": [
                {
                    "properties": {
                        "name": {
                            "type": "string",
                            "enum": ["AsyncHyperBandScheduler"]
                        },
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "time_attr": {
                                    "type": "string",
                                    "description": """A training result
                                     attr to use for comparing time.
                                     Note that you can pass in something
                                      non-temporal such as
                                      training_iteration as a measure of
                                    progress, the only requirement is that
                                     the attribute should increase
                                      monotonically.""",
                                    "enum": ["training_iteration"],
                                    "default": "training_iteration"
                                },
                                "max_t": {
                                    "type": "number",
                                    "description": """max time units per
                                     trial. Trials will be stopped after
                                     max_t time units (determined by
                                      time_attr) have passed.""",
                                    "default": 100.0
                                },
                                "grace_period": {
                                    "type": "number",
                                    "description": """Only stop trials at
                                    least this old in time. The units are
                                     the same as the attribute named by
                                     time_attr""",
                                    "default": 1.0
                                },
                                "reduction_factor": {
                                    "type": "number",
                                    "description": """Used to set halving
                                    rate and amount. This is simply a
                                    unit-less scalar.""",
                                    "default": 4.0
                                },
                                "brackets": {
                                    "type": "integer",
                                    "description": """Number of brackets.
                                    Each bracket has a different halving
                                    rate, specified by the reduction
                                    factor.""",
                                    "default": 1
                                }
                            }
                        }
                    }
                },
                {
                    "properties": {
                        "name": {
                            "type": "string",
                            "enum": ["HyperBandScheduler"]
                        },
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "time_attr": {
                                    "type": "string",
                                    "description": """A training result attr to
                                    use for comparing time. Note that you
                                    can pass in something non-temporal such
                                    as training_iteration as a measure of
                                    progress, the only requirement is that
                                     the attribute should increase
                                      monotonically.""",
                                    "enum": ["training_iteration"],
                                    "default": "training_iteration"
                                },
                                "max_t": {
                                    "type": "number",
                                    "description": """max time units per
                                     trial. Trials will be stopped after
                                     max_t time units (determined by
                                      time_attr) have passed.""",
                                    "default": 100.0
                                },
                                "reduction_factor": {
                                    "type": "number",
                                    "description": """Used to set halving
                                    rate and amount. This is simply a
                                    unit-less scalar.""",
                                    "default": 4.0
                                },
                            }
                        }
                    }
                },
                {
                    "properties": {
                        "name": {
                            "type": "string",
                            "enum": ["MedianStoppingRule"]
                        },
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "time_attr": {
                                    "type": "string",
                                    "description": """A training result attr to
                                    use for comparing time. Note that you
                                    can pass in something non-temporal such
                                    as training_iteration as a measure of
                                    progress, the only requirement is that
                                     the attribute should increase
                                      monotonically.""",
                                    "enum": ["training_iteration"],
                                    "default": "training_iteration"
                                },
                                "grace_period": {
                                    "type": "number",
                                    "description": """Only stop trials at
                                    least this old in time. The units are
                                     the same as the attribute named by
                                     time_attr""",
                                    "default": 60.0
                                },
                                "min_samples_required": {
                                    "type": "integer",
                                    "description": """Minimum number of
                                     trials to compute median over.""",
                                    "default": 3
                                },
                                "min_time_slice": {
                                    "type": "number",
                                    "description": """Each trial runs at
                                    least this long before yielding
                                     (assuming it isn’t stopped).
                                     Note: trials ONLY yield if there
                                    are not enough samples to evaluate
                                    performance for the current result
                                    AND there are other trials waiting to
                                    run. The units are the same as the
                                     attribute named by time_attr.""",
                                    "default": 0.0
                                },
                                "hard_stop": {
                                    "type": "boolean",
                                    "description": """If False, pauses
                                     trials instead of stopping them.
                                     When all other trials are complete,
                                     paused trials will be resumed and
                                     allowed to run FIFO.""",
                                    "default": True
                                },
                            }
                        }
                    }
                },
            ]
        }
    }
}


class NemoHyperTuneNode(GridRandomSearchNode):

    def init(self):
        GridRandomSearchNode.init(self)

    def ports_setup(self):
        return GridRandomSearchNode.ports_setup(self)

    def columns_setup(self):
        return GridRandomSearchNode.columns_setup(self)

    def conf_schema(self):
        cache_key, task_graph, _ = self._compute_hash_key()
        if cache_key in cache_schema:
            return cache_schema[cache_key]
        tensors = []
        if task_graph is not None:
            for task in task_graph:
                if task.get('type') == 'NemoTrainNode':
                    conf = task.get('conf')
                    if ('eval_callback' in conf and
                            'eval_tensors' in conf['eval_callback']):
                        tensors = conf['eval_callback']['eval_tensors']
                        tensors = [t.split('@')[-1] for t in tensors]
                        print(tensors)
        conf = GridRandomSearchNode.conf_schema(self)
        json = conf.json
        if 'properties' in json:
            del json['properties']['metrics']
            json['properties']['best'][
                'properties']['metric']['enum'] = tensors
            json['properties']['scheduler'] = copy.deepcopy(_SCHED_CONF)
        return ConfSchema(json=json, ui=conf.ui)

    def process(self, inputs):
        _, task_graph, _ = self._compute_hash_key()
        train_id = None
        if task_graph is not None:
            for task in task_graph:
                if task.get('type') == 'NemoTrainNode':
                    conf = task.get('conf')
                    if ('eval_callback' in conf and
                            'eval_tensors' in conf['eval_callback']):
                        tensors = conf['eval_callback']['eval_tensors']
                        tensors = [t.split('@')[-1] for t in tensors]
                        train_id = task.get('id')
        if train_id is None:
            print('no train node detected')
            return {}

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

                outputLists = [train_id+'.'+'checkpoint_dir']
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
                self._make_sub_graph_connection(task_graph,
                                                inputNode_fun, outNode_fun)

                task_graph.extend(input_feeders)
                self.update_conf_for_search(replaceObj, task_graph, config)
                task_graph.run(outputLists, replace=replaceObj)
                # metric_report = {item: result[item] for item in outputLists}
                # tune.report(**metric_report)
            config = {}
            for para in self.conf['parameters']:
                fun_name = para['search']['function']
                fun = getattr(tune, fun_name)
                if fun_name == 'grid_search' or fun_name == 'choice':
                    config[para['name']] = fun(para['search']['args'])
                else:
                    config[para['name']] = fun(*para['search']['args'])

            scheduler_instance = None
            if 'scheduler' in self.conf and 'name' in self.conf['scheduler']:
                import ray.tune.schedulers
                sconf = self.conf['scheduler']
                name = sconf['name']
                scheduler = getattr(ray.tune.schedulers, name)
                para = sconf['parameters']
                para.update(self.conf['best'])
                print(para)
                scheduler_instance = scheduler(**para)

            if scheduler_instance is None:
                analysis = tune.run(search_fun, **self.conf['tune'],
                                    config=config)
            else:
                analysis = tune.run(search_fun, **self.conf['tune'],
                                    config=config,
                                    scheduler=scheduler_instance)
            best = analysis.get_best_config(**self.conf['best'])
            print('best parameter', best)
            for key in best.keys():
                self.conf['context'][key]['value'] = best[key]
            output[self.OUTPUT_CONFIG] = self.conf

        # TODO: Fix the check point directory loading. Ray creates checkpoint
        #     directories under its "tune->local_dir". The directory names are
        #     taken from the taskgraph on which the HPO is being perfomed.
        #     These checkpoint directories within ray subdirectory need to
        #     override or deal with the checkpoint directories that might be
        #     set for the taskgraph for which the HPO is being performed.
        # print('NemoHyperTuneNode CONF:\n{}'.format(self.conf))

        more_output = ContextCompositeNode.process(self, inputs)
        output.update(more_output)

        return output
