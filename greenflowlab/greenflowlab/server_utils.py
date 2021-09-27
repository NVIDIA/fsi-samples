import inspect
import uuid
from greenflow.dataframe_flow import TaskGraph
from greenflow.dataframe_flow import Node
from greenflow.dataframe_flow.task import Task
from greenflow.dataframe_flow.output_collector_node import (
    OUTPUT_TYPE, OUTPUT_ID)
from greenflow.dataframe_flow import (TaskSpecSchema, PortsSpecSchema)
from greenflow.dataframe_flow.config_nodes_modules import (
    load_modules, get_greenflow_config_modules, get_node_tgraphmixin_instance)
import greenflow.plugin_nodes as plugin_nodes

try:
    # For python 3.8 and later
    import importlib.metadata as importlib_metadata
except ImportError:
    # prior to python 3.8 need to install importlib-metadata
    import importlib_metadata
from pathlib import Path

DYNAMIC_MODULES = {}


def register_node(module, classObj):
    if module not in DYNAMIC_MODULES:
        container = {}
        DYNAMIC_MODULES[module] = container
    else:
        container = DYNAMIC_MODULES[module]

    # Snippet below is important so that dynamic modules appear under the
    # "Add Nodes" menu.
    key = classObj.__name__
    container[key] = classObj


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
    dynamic = PortsSpecSchema.dynamic

    all_ports = []
    for key in port:
        one_port = {}
        one_port['name'] = key

        if dynamic in port[key]:
            one_port[dynamic] = port[key][dynamic]

        port_type = port[key][PortsSpecSchema.port_type]
        if isinstance(port_type, list):
            types = []
            for t in port_type:
                type_names = [e.__module__+'.'+e.__name__ for
                              e in t.mro()]
                types.append(type_names)
            one_port['type'] = types
        else:
            type_name = [e.__module__+'.'+e.__name__ for
                         e in port_type.mro()]
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
            input ports, output ports, output meta names/types,
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
            # Setting output collector ID should not be needed.
            task._task_spec[TaskSpecSchema.task_id] = OUTPUT_ID
            # task._task_spec[TaskSpecSchema.node_type] = Output_Collector
    task_graph.build()
    nodes = []
    edges = []
    for task in task_graph:
        node = task_graph[task[TaskSpecSchema.task_id]]
        out_node = get_labnode_obj(node)
        connection_inputs = task.get('inputs')
        nodes.append(out_node)
        # out_node['output_meta'] = task_graph[node.uid].output_meta
        for port, v in connection_inputs.items():
            edge = {"from": v, "to": node.uid+"."+port}
            edges.append(edge)
        # fix the output collector inputs
        if (task[TaskSpecSchema.task_id] == OUTPUT_ID):
            inputs = []
            num = 0
            for port, v in connection_inputs.items():
                inputs.append({'name': port, "type": [["any"]]})
                num = max(int(port[2:]), num)
            inputs.append({'name': 'in'+str(num+1), "type": [["any"]]})
            out_node['inputs'] = inputs

    task_graph.run_cleanup()

    return {'nodes': nodes, 'edges': edges}


def init_get_labnode_obj(node, count_id=True):
    node.init()
    node.update()
    return get_labnode_obj(node, count_id=count_id)


def get_labnode_obj(node, count_id=True):
    """
    It is a private function to convert a Node instance into a dictionary for
    client to consume.

    Arguments
    -------
    node: Node
        greenflow Node

    Returns
    -------
    dict
        node data for client
    """
    ports = node.ports_setup()
    metadata = node.meta_setup()
    schema = node.conf_schema()
    typeName = node._task_obj.get('type')
    if node.uid == OUTPUT_ID:
        width = 160
        typeName = OUTPUT_TYPE
    else:
        if count_id:
            width = max(max(len(node.uid), len(typeName)) * 10, 100)
        else:
            width = max(len(typeName) * 10, 100)

    conf = node._task_obj.get('conf')
    out_node = {'width': width,
                'id': node.uid,
                'type': typeName,
                'schema': schema.json,
                'ui': schema.ui,
                'conf': conf,
                'inputs': _format_port(ports.inports),
                'outputs': _format_port(ports.outports)}
    out_node['output_meta'] = metadata.outports

    if node._task_obj.get('filepath'):
        out_node['filepath'] = node._task_obj.get('filepath')

    if node._task_obj.get('module'):
        out_node['module'] = node._task_obj.get('module')

    out_node['required'] = metadata.inports
    return out_node


def get_nodes_from_file(file):
    """
    Given an input yaml file string. It returns a dict which has two keys.
        nodes:
            - list of node objects for the UI client. It contains all the
            necessary information about the node including the size of the node
            input ports, output ports, output meta names/types,
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
    to the graph. The nodes are from two sources: default greenflow nodes and
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
    loaded_node_classes = []
    all_modules = get_greenflow_config_modules()
    # print('Greenflow config modules: {}\n'.format(all_modules))
    all_nodes = {}
    # not implemented yet for greenflow
    for item in inspect.getmembers(plugin_nodes):
        if inspect.ismodule(item[1]):
            # print('Greenflow builtin plugin: {}'.format(item))
            labmod_pkg = 'greenflow.{}'.format(item[0])
            all_nodes[labmod_pkg] = []
            for node in inspect.getmembers(item[1]):
                nodecls = node[1]
                if not inspect.isclass(nodecls):
                    continue
                if not issubclass(nodecls, Node):
                    continue
                if nodecls in loaded_node_classes:
                    continue

                task = {'id': 'node_'+str(uuid.uuid4()),
                        'type': node[0],
                        'conf': {},
                        'inputs': []}
                task_inst = Task(task)
                node_inst = get_node_tgraphmixin_instance(nodecls, task_inst)
                nodeObj = init_get_labnode_obj(node_inst, False)
                all_nodes[labmod_pkg].append(nodeObj)
                loaded_node_classes.append(nodecls)

    for module in all_modules:
        module_file_or_path = Path(all_modules[module])
        loaded = load_modules(all_modules[module], module)
        mod = loaded.mod
        modulename = module

        # all_nodes[modulename] = []
        for node in inspect.getmembers(mod):
            nodecls = node[1]
            if not inspect.isclass(nodecls):
                continue
            if nodecls == Node:
                continue

            if not issubclass(nodecls, Node):
                continue

            if nodecls in loaded_node_classes:
                continue

            task = {'id': 'node_'+str(uuid.uuid4()),
                    'type': node[0],
                    'conf': {},
                    'inputs': [],
                    'module': module
                    }
            task_inst = Task(task)
            node_inst = get_node_tgraphmixin_instance(nodecls, task_inst)
            nodeObj = init_get_labnode_obj(node_inst, False)
            if module_file_or_path.is_dir():
                # submod = nodecls.__module__.split('.')[1:]
                # flatten out the namespace hierarchy
                submod = nodecls.__module__.split('.')[1:2]
                modulename_ = '.'.join([modulename, '.'.join(submod)]) \
                    if submod else modulename
                all_nodes.setdefault(modulename_, []).append(nodeObj)
            else:
                all_nodes.setdefault(modulename, []).append(nodeObj)

            loaded_node_classes.append(nodecls)

    for module in DYNAMIC_MODULES.keys():
        modulename = module
        all_nodes[modulename] = []
        for class_name in DYNAMIC_MODULES[module].keys():
            nodecls = DYNAMIC_MODULES[module][class_name]

            if not issubclass(nodecls, Node):
                continue

            if nodecls in loaded_node_classes:
                continue

            task = {'id': 'node_'+str(uuid.uuid4()),
                    'type': nodecls.__name__,
                    'conf': {},
                    'inputs': [],
                    'module': module
                    }
            task_inst = Task(task)
            node_inst = get_node_tgraphmixin_instance(nodecls, task_inst)
            nodeObj = init_get_labnode_obj(node_inst, False)
            all_nodes.setdefault(modulename, []).append(nodeObj)
            loaded_node_classes.append(nodecls)

    # load all the plugins from entry points
    for entry_point in importlib_metadata.entry_points().get(
            'greenflow.plugin', ()):
        mod = entry_point.load()
        modulename = entry_point.name

        for node in inspect.getmembers(mod):
            nodecls = node[1]
            if not inspect.isclass(nodecls):
                continue
            if nodecls == Node:
                continue

            if not issubclass(nodecls, Node):
                continue

            if nodecls in loaded_node_classes:
                continue

            task = {'id': 'node_'+str(uuid.uuid4()),
                    'type': node[0],
                    'conf': {},
                    'inputs': [],
                    'module': modulename
                    }
            task_inst = Task(task)
            node_inst = get_node_tgraphmixin_instance(nodecls, task_inst)
            nodeObj = init_get_labnode_obj(node_inst, False)
            all_nodes.setdefault(modulename, []).append(nodeObj)
            loaded_node_classes.append(nodecls)

    return all_nodes
