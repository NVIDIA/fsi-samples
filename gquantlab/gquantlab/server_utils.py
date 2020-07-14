from gquant.dataframe_flow import TaskGraph
from gquant.dataframe_flow import Node
from gquant.dataframe_flow.task import Task
import importlib
import pathlib
import gquant.plugin_nodes as plugin_nodes
import inspect
from gquant.dataframe_flow import Node
import uuid
from datetime import datetime as dt


def format_port(port):
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
    width = max(max(len(node.uid), len(typeName)) * 10, 100)
    conf = node._task_obj.get('conf')
    out_node = {'width': width,
                'id': node.uid,
                'type': typeName,
                'schema': schema.json,
                'ui': schema.ui,
                'conf': conf,
                'inputs': format_port(ports.inports),
                'outputs': format_port(ports.outports)}
    out_node['required'] = node.required
    out_node['output_columns'] = {}
    if node._task_obj.get('filepath'):
        out_node['filepath'] = node._task_obj.get('filepath')
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
    task_graph.build()
    nodes = []
    edges = []
    for task in task_graph:
        node = task.get_node_obj()
        out_node = get_node_obj(node)
        connection_inputs = task.get('inputs')
        nodes.append(out_node)
        out_node['output_columns'] = task_graph[node.uid].output_columns
        for port, v in connection_inputs.items():
            edge = {"from": v, "to": node.uid+"."+port}
            edges.append(edge)

    return {'nodes': nodes, 'edges': edges}


def add_nodes():
    """
    It will load all the nodes for the UI client so user can add new node 
    to the graph. The nodes are from two sources: 
    default gQuant nodes and customized node modules. Currently gQuant default
    nodes are still using old API. So all the nodes are coming from customized 
    node modules in the 'modules' directory.
    
    The output is a dictionary whose keys are module names and values are a
    list of the nodes inside that module. 

    Arguments
    -------

    Returns
    -------
    dict
        dictionary of all the nodes that can be added in the client
    """
    all_modules = load_all_modules()
    # not implemented yet for gQuant
    # for item in inspect.getmembers(plugin_nodes):
    #     if inspect.ismodule(item[1]):
    #         print(item)
    #         for node in inspect.getmembers(item[1]):
    #             if inspect.isclass(node[1]):
    #                 task = {'id': 'random',
    #                         'type': node[0],
    #                         'conf': {},
    #                         'inputs': []}
    #                 t = Task(task)
    #                 n = node[1](t)
    #                 if issubclass(node[1], Node):
    #                     print(get_node_obj(n))
    all_nodes = {}
    for module in all_modules:
        mod = importlib.import_module(module)
        all_nodes[module] = []
        for node in inspect.getmembers(mod):
            if node[1] == Node:
                continue
            if inspect.isclass(node[1]):
                if issubclass(node[1], Node):
                    task = {'id': 'node_'+str(uuid.uuid4())[:8],
                            'type': node[0],
                            'conf': {},
                            'inputs': [],
                            'filepath': 'modules/'+module+'.py'
                            }
                    t = Task(task)
                    n = node[1](t)
                    nodeObj = get_node_obj(n)
                    all_nodes[module].append(nodeObj)
    return all_nodes


def load_all_modules():
    """
    This is a utility function to load all the customized module files
    return list of modules

    Returns
    -------
    list
        list of moudles in the 'modules' directory
    """
    all_modules = pathlib.Path('modules').glob('*.py')
    return [module.stem for module in all_modules]
