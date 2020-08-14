from gquant.dataframe_flow import Node
from gquant.dataframe_flow import TaskGraph
from gquant.dataframe_flow.taskSpecSchema import TaskSpecSchema
from gquant.dataframe_flow.portsSpecSchema import ConfSchema
from gquant.dataframe_flow.portsSpecSchema import NodePorts
from gquant.dataframe_flow.cache import (cache_columns,
                                         cache_ports, cache_schema)
import json
import os
import hashlib
from gquant.dataframe_flow.util import get_file_path
import uuid
import copy

INPUT_ID = '4fd31358-fb80-4224-b35f-34402c6c3763'


def _get_node(port_name):
    return '@'.join(port_name.split('@')[:-1])


def _get_port(port_name):
    return port_name.split('@')[-1] 


def fix_port_name(obj, subgraph_node_name):
    output = {}
    for key in obj.keys():
        output[subgraph_node_name+'@'+key] = obj[key]
    return output


class CompositeNode(Node):

    def _compute_hash_key(self):
        """
        if hash changed, the port_setup, columns_setup
        and conf_json should be different
        """
        task_graph = ""
        inputs = ()
        replacementObj = {}
        input_node = ""
        self.update_replace(replacementObj)
        if 'taskgraph' in self.conf:
            task_graph = get_file_path(self.conf['taskgraph'])
            if os.path.exists(task_graph):
                with open(task_graph) as f:
                    task_graph = hashlib.md5(f.read().encode()).hexdigest()
        if 'input' in self.conf:
            for inp in self.conf['input']:
                input_node += inp+","
                if hasattr(self, 'inputs'):
                    for i in self.inputs:
                        inputs += (hash(i['from_node']),
                                   i['to_port'], i['from_port'])
        return hash((self.uid, task_graph, inputs,
                     input_node, json.dumps(replacementObj)))

    def _make_sub_graph_connection(self, task_graph,
                                   inputNode_fun,
                                   outNode_fun):

        """
        connects the current composite node's inputs and outputs to 
        the subgraph-task_graph's inputs and outputs. 
        inputNode_fun has subgraph inputNode as argument, 
        it processes the inputNode logics
        outputNode_fun has subgraph outputNode as argument, 
        it processes the outNode logics
        """
        if 'input' in self.conf:
            for inp in self.conf['input']:
                if inp in task_graph:
                    inputNode = task_graph[inp]
                    inputNode.inputs.clear()
                    if hasattr(self, 'inputs'):
                        for currentInput in self.inputs:
                            if _get_node(
                                    currentInput[
                                        'to_port']) == inputNode.uid:
                                # change the input name
                                newInput = {}
                                newInput['to_port'] = _get_port(
                                    currentInput['to_port'])
                                newInput['from_port'] = currentInput['from_port']
                                newInput['from_node'] = currentInput['from_node']
                                inputNode.inputs.append(newInput)
                    inputNode_fun(inputNode)
                    # required.update(fix_port_name(inputNode.required,
                    #                               inputNode.uid))
                    # inports.update(fix_port_name(
                    #     inputNode.ports_setup().inports, inputNode.uid))
        if 'output' in self.conf:
            for oup in self.conf['output']:
                if oup in task_graph:
                    outNode = task_graph[oup]
                    outNode.outputs.clear()
                    if hasattr(self, 'outputs'):
                        for currentOutput in self.outputs:
                            if _get_node(
                                currentOutput[
                                    'from_port']) == outNode.uid:
                                # change the input name
                                newOutput = {}
                                newOutput['from_port'] = _get_port(
                                    currentOutput['from_port'])
                                newOutput['to_port'] = currentOutput['to_port']
                                newOutput['to_node'] = currentOutput['to_node']
                                outNode.outputs.append(newOutput)
                    outNode_fun(outNode)
                    # outports.update(fix_port_name(
                    #     outNode.ports_setup().outports, outNode.uid))

    def ports_setup(self):
        cache_key = self._compute_hash_key()
        if cache_key in cache_ports:
            # print('cache hit')
            return cache_ports[cache_key]
        required = {}
        inports = {}
        outports = {}
        if 'taskgraph' in self.conf:
            task_graph = TaskGraph.load_taskgraph(
                get_file_path(self.conf['taskgraph']))
            replacementObj = {}
            self.update_replace(replacementObj)
            task_graph.build(replace=replacementObj)

            def inputNode_fun(inputNode):
                required.update(fix_port_name(inputNode.required,
                                              inputNode.uid))
                inports.update(fix_port_name(
                    inputNode.ports_setup().inports, inputNode.uid))

            def outNode_fun(outNode):
                outports.update(fix_port_name(
                    outNode.ports_setup().outports, outNode.uid))

            self._make_sub_graph_connection(task_graph,
                                            inputNode_fun, outNode_fun)
        self.required = required
        output_port = NodePorts(inports=inports, outports=outports)
        cache_ports[cache_key] = output_port
        return output_port

    def columns_setup(self):
        cache_key = self._compute_hash_key()
        if cache_key in cache_columns:
            # print('cache hit')
            return cache_columns[cache_key]
        out_columns = {}
        if 'taskgraph' in self.conf:
            task_graph = TaskGraph.load_taskgraph(
                get_file_path(self.conf['taskgraph']))
            replacementObj = {}
            self.update_replace(replacementObj)
            task_graph.build(replace=replacementObj)

            def inputNode_fun(inputNode):
                pass

            def outNode_fun(outNode):
                out_columns.update(fix_port_name(outNode.columns_setup(),
                                                 outNode.uid))

            self._make_sub_graph_connection(task_graph,
                                            inputNode_fun, outNode_fun)

        cache_columns[cache_key] = out_columns
        return out_columns

    def conf_schema(self):
        cache_key = self._compute_hash_key()
        if cache_key in cache_schema:
            # print('cache hit')
            return cache_schema[cache_key]
        json = {
            "title": "Composite Node configure",
            "type": "object",
            "description": """Use a sub taskgraph as a composite node""",
            "properties": {
                "taskgraph":  {
                    "type": "string",
                    "description": "the taskgraph filepath"
                },
                "input":  {
                    "type": "array",
                    "description": "the input node ids",
                    "items": {
                        "type": "string"
                    }
                },
                "output":  {
                    "type": "array",
                    "description": "the output node ids",
                    "items": {
                        "type": "string"
                    }
                },
                "subnode_ids":  {
                    "title": self.uid+" subnode ids",
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": """sub graph node ids that need
                    to be reconfigured"""
                },
                "subnodes_conf":  {
                    "title": self.uid+" subnodes configuration",
                    "type": "object",
                    "properties": {}
                }
            },
            "required": ["taskgraph"],
        }
        ui = {
            "taskgraph": {"ui:widget": "TaskgraphSelector"},
            "subnodes_conf": {}
        }
        if 'taskgraph' in self.conf:
            task_graphh = TaskGraph.load_taskgraph(
                get_file_path(self.conf['taskgraph']))
            replacementObj = {}
            self.update_replace(replacementObj)
            task_graphh.build(replace=replacementObj)

            def inputNode_fun(inputNode):
                pass

            def outNode_fun(outNode):
                pass

            self._make_sub_graph_connection(task_graphh,
                                            inputNode_fun, outNode_fun)

            ids_in_graph = []
            for t in task_graphh:
                ids_in_graph.append(t.get('id'))
            json['properties']['input']['items']['enum'] = ids_in_graph
            json['properties']['output']['items']['enum'] = ids_in_graph
            json['properties']['subnode_ids']['items']['enum'] = ids_in_graph
        if 'subnode_ids' in self.conf:
            for subnodeId in self.conf['subnode_ids']:
                if subnodeId in task_graphh:
                    nodeObj = task_graphh[subnodeId]
                    schema = nodeObj.conf_schema()
                    json['properties'][
                        "subnodes_conf"]['properties'][subnodeId] = {
                            "type": "object",
                            "properties": {
                                "conf": schema.json
                            }
                    }
                    ui["subnodes_conf"].update({
                        subnodeId: {
                            'conf': schema.ui
                        }
                    })
        out_schema = ConfSchema(json=json, ui=ui)
        cache_schema[cache_key] = out_schema
        return out_schema

    def update_replace(self, replaceObj):
        # find the other replacment conf
        if 'subnodes_conf' in self.conf:
            for key in self.conf['subnodes_conf'].keys():
                newid = key
                if newid in replaceObj:
                    replaceObj[newid].update(self.conf[
                        'subnodes_conf'][key])
                else:
                    replaceObj[newid] = {}
                    replaceObj[newid].update(self.conf[
                        'subnodes_conf'][key])

    def process(self, inputs):
        """
        Composite computation

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        if 'taskgraph' in self.conf:
            task_graph = TaskGraph.load_taskgraph(
                get_file_path(self.conf['taskgraph']))
            task_graph.build()

            outputLists = []
            replaceObj = {}
            input_feeders = []

            def inputNode_fun(inputNode):
                inports = inputNode.ports_setup().inports

                class InputFeed(Node):

                    def columns_setup(self2):
                        output = {}
                        if hasattr(self, 'inputs'):
                            # inputNode.inputs.extend(self.inputs)
                            for inp in inputNode.inputs:
                                output[inp['to_port']] = inp['from_node'].columns_setup()[inp['from_port']]
                        # it will be something like { input_port: columns }
                        return output 

                    def ports_setup(self2):
                        # it will be something like { input_port: types }
                        return NodePorts(inports={}, outports=inports)

                    def conf_schema(self2):
                        return ConfSchema()

                    def process(self2, empty):
                        output = {}
                        for key in inports.keys():
                            if inputNode.uid+'@'+key in inputs:
                                output[key] = inputs[inputNode.uid+'@'+key]
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
                    if inputNode.uid+'@'+key in inputs:
                        newInputs[key] = uni_id+'.'+key
                replaceObj.update({inputNode.uid: {
                    TaskSpecSchema.inputs: newInputs}
                })

            def outNode_fun(outNode):
                out_ports = outNode.ports_setup().outports
                # fixed_outports = fix_port_name(out_ports, outNode.uid)
                for key in out_ports.keys():
                    if self.outport_connected(outNode.uid+'@'+key):
                        outputLists.append(outNode.uid+'.'+key)

            self._make_sub_graph_connection(task_graph,
                                            inputNode_fun, outNode_fun)

            task_graph.extend(input_feeders)
            self.update_replace(replaceObj)
            result = task_graph.run(outputLists, replace=replaceObj)
            output = {}
            for key in result.get_keys():
                splits = key.split('.') 
                output['@'.join(splits)] = result[key]
            return output
        else:
            return {}
