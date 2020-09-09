from gquant.dataframe_flow import TaskGraph
from gquant.dataframe_flow import Node
from gquant.dataframe_flow.task import Task
from gquant.dataframe_flow._node_flow import OUTPUT_TYPE, OUTPUT_ID
from gquant.dataframe_flow import TaskSpecSchema
from gquant.dataframe_flow.task import load_modules, get_gquant_config_modules
import gquant.plugin_nodes as plugin_nodes
import inspect
import uuid
# import sys
# sys.path.append('modules') # noqa E262


def _format_port(port):
    """
    compute the right port type str

    Arguments
    -------
    port: input/output port object

    Returns
    -------
    list
        a list of ports with name and type
    """
    all_ports = []
    for key in port:
        one_port = {}
        one_port['name'] = key
        port_type = port[key]['type']
        if isinstance(port_type, list):
            types = []
            for t in port_type:
                type_name = t.__module__+'.'+t.__name__
                types.append(type_name)
            one_port['type'] = types
        else:
            type_name = port_type.__module__+'.'+port_type.__name__
            one_port['type'] = [type_name]
        all_ports.append(one_port)
    return all_ports


def get_nodes(task_graph):
    """
    It is a private function taking an input task graph. It will run the
    column flow and compute the column names and types for all the nodes.

    It returns a dict which has two keys.
        nodes:
            - list of node objects for the UI client. It contains all the
            necessary information about the node including the size of the node
            input ports, output ports, output column names/types,
            conf schema and conf data.
        edges:
            - list of edge objects for the UI client. It enumerate all the
            edges in the graph.

    Arguments
    -------
    task_graph: TaskGraph
        taskgraph object

    Returns
    -------
    dict
        nodes and edges of the graph data
    """
    for task in task_graph:
        if (task.get(TaskSpecSchema.node_type) == OUTPUT_TYPE):
            task.set_output()
    task_graph.build()
    nodes = []
    edges = []
    for task in task_graph:
        # node = task.get_node_obj()
        node = task_graph[task[TaskSpecSchema.task_id]]
        out_node = get_node_obj(node)
        connection_inputs = task.get('inputs')
        nodes.append(out_node)
        # out_node['output_columns'] = task_graph[node.uid].output_columns
        for port, v in connection_inputs.items():
            edge = {"from": v, "to": node.uid+"."+port}
            edges.append(edge)
        # fix the output collector inputs
        if (task[TaskSpecSchema.task_id] == OUTPUT_ID):
            inputs = []
            num = 0
            for port, v in connection_inputs.items():
                inputs.append({'name': port, "type": ["any"]})
                num = max(int(port[2:]), num)
            inputs.append({'name': 'in'+str(num+1), "type": ["any"]})
            out_node['inputs'] = inputs
    return {'nodes': nodes, 'edges': edges}


def get_node_obj(node):
    """
    It is a private function to convert a Node instance into a dictionary for
    client to consume.

    Arguments
    -------
    node: Node
        gquant Node

    Returns
    -------
    dict
        node data for client
    """
    ports = node.ports_setup()
    schema = node.conf_schema()
    typeName = node._task_obj.get('type')
    if node.uid == OUTPUT_ID:
        width = 160
        typeName = OUTPUT_TYPE
    else:
        width = max(max(len(node.uid), len(typeName)) * 10, 100)
    conf = node._task_obj.get('conf')
    out_node = {'width': width,
                'id': node.uid,
                'type': typeName,
                'schema': schema.json,
                'ui': schema.ui,
                'conf': conf,
                'inputs': _format_port(ports.inports),
                'outputs': _format_port(ports.outports)}
    out_node['output_columns'] = node.columns_setup()
    if node._task_obj.get('filepath'):
        out_node['filepath'] = node._task_obj.get('filepath')
    if node._task_obj.get('module'):
        out_node['module'] = node._task_obj.get('module')
    out_node['required'] = node.required
    return out_node


def get_nodes_from_file(file):
    """
    Given an input yaml file string. It returns a dict which has two keys.
        nodes:
            - list of node objects for the UI client. It contains all the
            necessary information about the node including the size of the node
            input ports, output ports, output column names/types,
            conf schema and conf data.
        edges:
            - list of edge objects for the UI client. It enumerate all the
            edges in the graph.

    Arguments
    -------
    file: string
        file name

    Returns
    -------
    dict
        nodes and edges of the graph data

    """
    task_graph = TaskGraph.load_taskgraph(file)
    return get_nodes(task_graph)


def add_nodes():
    """
    It will load all the nodes for the UI client so user can add new node
    to the graph. The nodes are from two sources: default gQuant nodes and
    customized node modules.

    The output is a dictionary whose keys are module names and values are a
    list of the nodes inside that module.

    Arguments
    -------

    Returns
    -------
    dict
        dictionary of all the nodes that can be added in the client
    """
    all_modules = get_gquant_config_modules()
    print(all_modules)
    all_nodes = {}
    # not implemented yet for gQuant
    for item in inspect.getmembers(plugin_nodes):
        if inspect.ismodule(item[1]):
            print(item)
            all_nodes[item[0]] = []
            for node in inspect.getmembers(item[1]):
                if inspect.isclass(node[1]):
                    task = {'id': 'random',
                            'type': node[0],
                            'conf': {},
                            'inputs': []}
                    t = Task(task)
                    n = node[1](t)
                    if issubclass(node[1], Node):
                        nodeObj = get_node_obj(n)
                        all_nodes[item[0]].append(nodeObj)
    for module in all_modules:
        loaded = load_modules(all_modules[module], module)
        mod = loaded.mod
        modulename = module

        all_nodes[modulename] = []
        for node in inspect.getmembers(mod):
            if node[1] == Node:
                continue
            if inspect.isclass(node[1]):
                if issubclass(node[1], Node):
                    task = {'id': 'node_'+str(uuid.uuid4())[:8],
                            'type': node[0],
                            'conf': {},
                            'inputs': [],
                            'module': module
                            }
                    t = Task(task)
                    n = node[1](t)
                    nodeObj = get_node_obj(n)
                    all_nodes[modulename].append(nodeObj)
    return all_nodes
