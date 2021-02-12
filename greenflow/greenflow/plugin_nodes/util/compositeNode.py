from greenflow.dataframe_flow import Node
from greenflow.dataframe_flow import TaskGraph
from greenflow.dataframe_flow.taskSpecSchema import TaskSpecSchema
from greenflow.dataframe_flow.portsSpecSchema import ConfSchema
from greenflow.dataframe_flow.portsSpecSchema import NodePorts, MetaData
from greenflow.dataframe_flow.cache import (CACHE_META,
                                         CACHE_PORTS, CACHE_SCHEMA)
import json
import os
import hashlib
from greenflow.dataframe_flow.util import get_file_path
import uuid


__all__ = ["CompositeNode"]


def _get_node(port_name):
    return port_name.split('@')[0]


def _get_port(port_name):
    return '@'.join(port_name.split('@')[1:])


def fix_port_name(obj, subgraph_node_name):
    output = {}
    for key in obj.keys():
        output[subgraph_node_name+'@'+key] = obj[key]
    return output


def group_ports(input_list):
    """
    group inputs ports by node id
    returns a dictionary, keys are node id
    values are list of ports
    """
    nodes_group = {}
    for inp_port in input_list:
        inp = inp_port.split('.')[0]  # node id
        port_name = inp_port.split('.')[1]  # port name
        if inp in nodes_group:
            port_list = nodes_group.get(inp)
        else:
            port_list = []
            nodes_group[inp] = port_list
        port_list.append(port_name)
    return nodes_group


class CompositeNode(Node):

    def update(self):
        self.conf_update()  # update the conf

    def conf_update(self):
        """
        This method is used to overwrite the conf from
        external sources
        """
        pass

    def _compute_hash_key(self):
        """
        if hash changed, the port_setup, meta_setup
        and conf_json should be different
        In very rara case, might have the problem of hash collision,
        It affects the column, port and conf calculation. It won't
        change the computation result though.
        It returns the hash code, the loaded task_graph,
        the replacement conf obj
        """
        task_graph = ""
        inputs = ()
        replacementObj = {}
        input_node = ""
        task_graph_obj = None
        if 'taskgraph' in self.conf:
            try:
                task_graph = get_file_path(self.conf['taskgraph'])
            except FileNotFoundError:
                task_graph = None
            if task_graph is not None and os.path.exists(task_graph):
                with open(task_graph) as f:
                    task_graph = hashlib.md5(f.read().encode()).hexdigest()
                task_graph_obj = TaskGraph.load_taskgraph(
                    get_file_path(self.conf['taskgraph']))
        self.update_replace(replacementObj, task_graph_obj)
        if 'input' in self.conf:
            for inp in self.conf['input']:
                input_node += inp+","
                if hasattr(self, 'inputs'):
                    for i in self.inputs:
                        inputs += (hash(i['from_node']),
                                   i['to_port'], i['from_port'])
        return (hash((self.uid, task_graph, inputs, json.dumps(self.conf),
                      input_node, json.dumps(replacementObj))), task_graph_obj,
                replacementObj)

    def _make_sub_graph_connection(self, task_graph,
                                   inputNode_fun,
                                   outNode_fun):
        """
        connects the current composite node's inputs and outputs to
        the subgraph-task_graph's inputs and outputs.
        inputNode_fun has subgraph inputNode and all the input ports
        as argument, it processes the inputNode logics
        outputNode_fun has subgraph outputNode and all the outpout ports
        as argument, it processes the outNode logics
        """
        if 'input' in self.conf:
            # group input ports by node id
            inp_groups = group_ports(self.conf['input'])
            for inp in inp_groups.keys():
                if inp in task_graph:
                    inputNode = task_graph[inp]
                    update_inputs = []
                    replaced_ports = set(inp_groups[inp])
                    for oldInput in inputNode.inputs:
                        if oldInput['to_port'] in replaced_ports:
                            # we want to disconnect this old one and
                            # connect to external node
                            if hasattr(self, 'inputs'):
                                for externalInput in self.inputs:
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
                    inputNode_fun(inputNode, inp_groups[inp])

        if 'output' in self.conf:
            oup_groups = group_ports(self.conf['output'])
            for oup in oup_groups.keys():
                if oup in task_graph:
                    outNode = task_graph[oup]
                    # we do not disconnect anything here, as we take extra
                    # outputs for composite node.
                    # Node, we rely on the fact that taskgraph.run method
                    # will remove the output collector from taskgraph if
                    # the outputlist is set
                    outNode_fun(outNode, oup_groups[oup])

    def ports_setup(self):
        cache_key, task_graph, replacementObj = self._compute_hash_key()
        if cache_key in CACHE_PORTS:
            # print('cache hit')
            return CACHE_PORTS[cache_key]
        inports = {}
        outports = {}
        if task_graph:
            task_graph.build(replace=replacementObj)

            def inputNode_fun(inputNode, in_ports):
                inport = {}
                before_fix = inputNode.ports_setup().inports
                for key in before_fix.keys():
                    if key in in_ports:
                        inport[key] = before_fix[key]
                inports.update(fix_port_name(inport, inputNode.uid))

            def outNode_fun(outNode, out_ports):
                ouport = {}
                before_fix = outNode.ports_setup().outports
                for key in before_fix.keys():
                    if key in out_ports:
                        ouport[key] = before_fix[key]
                outports.update(fix_port_name(ouport, outNode.uid))

            self._make_sub_graph_connection(task_graph,
                                            inputNode_fun, outNode_fun)
        output_port = NodePorts(inports=inports, outports=outports)
        CACHE_PORTS[cache_key] = output_port
        return output_port

    def meta_setup(self):
        cache_key, task_graph, replacementObj = self._compute_hash_key()
        if cache_key in CACHE_META:
            # print('cache hit')
            return CACHE_META[cache_key]
        required = {}
        out_meta = {}
        if task_graph:
            task_graph.build(replace=replacementObj)

            def inputNode_fun(inputNode, in_ports):
                req = {}
                # do meta_setup so required columns are ready
                input_meta = inputNode.meta_setup().inports
                for key in input_meta.keys():
                    if key in in_ports:
                        req[key] = input_meta[key]
                required.update(fix_port_name(req, inputNode.uid))

            def outNode_fun(outNode, out_ports):
                oucols = {}
                before_fix = outNode.meta_setup().outports
                for key in before_fix.keys():
                    if key in out_ports:
                        oucols[key] = before_fix[key]
                out_meta.update(fix_port_name(oucols,
                                              outNode.uid))

            self._make_sub_graph_connection(task_graph,
                                            inputNode_fun, outNode_fun)
        metadata = MetaData(inports=required, outports=out_meta)
        CACHE_META[cache_key] = metadata
        return metadata

    def conf_schema(self):
        cache_key, task_graph, replacementObj = self._compute_hash_key()
        if cache_key in CACHE_SCHEMA:
            # print('cache hit')
            return CACHE_SCHEMA[cache_key]
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
        if task_graph:
            task_graph.build(replace=replacementObj)

            def inputNode_fun(inputNode, in_ports):
                pass

            def outNode_fun(outNode, out_ports):
                pass

            self._make_sub_graph_connection(task_graph,
                                            inputNode_fun, outNode_fun)

            ids_in_graph = []
            in_ports = []
            out_ports = []
            for t in task_graph:
                node_id = t.get('id')
                if node_id != '':
                    node = task_graph[node_id]
                    all_ports = node.ports_setup()
                    for port in all_ports.inports.keys():
                        in_ports.append(node_id+'.'+port)
                    for port in all_ports.outports.keys():
                        out_ports.append(node_id+'.'+port)
                    ids_in_graph.append(node_id)
            json['properties']['input']['items']['enum'] = in_ports
            json['properties']['output']['items']['enum'] = out_ports
            json['properties']['subnode_ids']['items']['enum'] = ids_in_graph
        if 'subnode_ids' in self.conf and task_graph:
            for subnodeId in self.conf['subnode_ids']:
                if subnodeId in task_graph:
                    nodeObj = task_graph[subnodeId]
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
        CACHE_SCHEMA[cache_key] = out_schema
        return out_schema

    def update_replace(self, replaceObj, task_graph=None):
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

            def inputNode_fun(inputNode, in_ports):
                inports = inputNode.ports_setup().inports

                class InputFeed(Node):

                    def meta_setup(self):
                        output = {}
                        for inp in inputNode.inputs:
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

                    def process(self, empty):
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
                for inp in inputNode.inputs:
                    if inp['to_port'] not in in_ports:
                        # need to keep the old connections
                        newInputs[inp['to_port']] = (inp['from_node'].uid
                                                     + '.' + inp['from_port'])
                replaceObj.update({inputNode.uid: {
                    TaskSpecSchema.inputs: newInputs}
                })

            def outNode_fun(outNode, out_ports):
                out_ports = outNode.ports_setup().outports
                # fixed_outports = fix_port_name(out_ports, outNode.uid)
                for key in out_ports.keys():
                    if self.outport_connected(outNode.uid+'@'+key):
                        outputLists.append(outNode.uid+'.'+key)

            self._make_sub_graph_connection(task_graph,
                                            inputNode_fun, outNode_fun)

            task_graph.extend(input_feeders)
            self.update_replace(replaceObj, task_graph)
            result = task_graph.run(outputLists, replace=replaceObj)
            output = {}
            for key in result.get_keys():
                splits = key.split('.')
                output['@'.join(splits)] = result[key]
            return output
        else:
            return {}
