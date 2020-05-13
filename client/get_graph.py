from gquant.dataframe_flow import TaskGraph
from gquant.dataframe_flow import Node
from gquant.dataframe_flow.task import Task
import importlib
import gquant.plugin_nodes as plugin_nodes
import inspect
from gquant.dataframe_flow import Node
import uuid


def get_node_obj(node):
    ports = node.ports_setup()
    schema = node.conf_schema()
    width = max(len(node.uid) * 10, 100)
    type = node._task_obj.get('type')
    conf = node._task_obj.get('conf')
    out_node = {'width': width,
                'id': node.uid,
                'type': type,
                'schema': schema.json,
                'ui': schema.ui,
                'conf': conf,
                'inputs': list(ports.inports.keys()),
                'outputs': list(ports.outports.keys())}
    out_node['required'] = node.required
    out_node['output_columns'] = {}
    if node._task_obj.get('filepath'):
        out_node['filepath'] = node._task_obj.get('filepath')
    return out_node


def get_nodes_from_file(file):
    task_graph = TaskGraph.load_taskgraph(file)
    return get_nodes(task_graph)


def get_nodes(task_graph):
    task_graph.build()
    nodes = []
    edges = []
    for task in task_graph:
        node = task.get_node_obj()
        # ports = node.ports_setup()
        # uid = node.uid

        # width = max(len(uid) * 10, 100)

        # out_node = {'width': width,
        #             'id': uid,
        #             'inputs': list(ports.inports.keys()),
        #             'outputs': list(ports.outports.keys())}
        out_node = get_node_obj(node)
        connection_inputs = task.get('inputs')
        nodes.append(out_node)
        out_node['output_columns'] = task_graph[node.uid].output_columns
        for port, v in connection_inputs.items():
            edge = {"from": v, "to": node.uid+"."+port}
            edges.append(edge)

    return {'nodes': nodes, 'edges': edges}


def add_nodes(module):
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
                        'filepath': module+'.py'
                        }
                t = Task(task)
                n = node[1](t)
                nodeObj = get_node_obj(n)
                all_nodes[module].append(nodeObj)
    return all_nodes