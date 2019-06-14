from collections import OrderedDict
import yaml
import copy
import networkx as nx
import importlib

from .node import Node

__all__ = ['run', 'save_workflow', 'load_workflow', 'viz_graph', 'get_graph']


DEFAULT_MODULE = "gquant.plugin_nodes"
mod_lib = importlib.import_module(DEFAULT_MODULE)

__SETUP_YAML_ONCE = False


def setup_yaml():
    '''Write out yaml in order for OrderedDict.'''
    # https://stackoverflow.com/a/8661021
    # https://stackoverflow.com/questions/47692094/lambda-works-defined-function-does-not  # noqa
    # represent_dict_order = lambda dumper, data: \
    #     dumper.represent_mapping('tag:yaml.org,2002:map', data.items())
    def represent_dict_order(dumper, data):
        return dumper.represent_mapping('tag:yaml.org,2002:map', data.items())
    yaml.add_representer(OrderedDict, represent_dict_order)

    global __SETUP_YAML_ONCE
    __SETUP_YAML_ONCE = True


def save_workflow(task_list, filename):
    """
    Write a list of tasks i.e. workflow to a yaml file.

    Arguments
    -------
    task_list: list
        List of dictionary objects describing tasks.
    filename: str
        The filename to write a yaml file to.

    """

    global __SETUP_YAML_ONCE
    if not __SETUP_YAML_ONCE:
        setup_yaml()

    # we want -id to be first in the resulting yaml file.
    tlist_od = []  # task list ordered
    for task in task_list:
        tod = OrderedDict([('id', 'idholder'), ('type', 'typeholder')])
        tod.update(task)
        tlist_od.append(tod)

    with open(filename, 'w') as fh:
        # yaml.dump(tlist_od, fh, default_flow_style=False, sort_keys=False)
        yaml.dump(tlist_od, fh, default_flow_style=False)


def load_workflow(filename):
    """
    load the yaml file to Python objects

    Arguments
    -------
    filename: str
        the filename pointing to the yaml file in the filesystem
    Returns
    -----
    object
        the Python objects representing the nodes in the graph

    """

    with open(filename) as f:
        obj = yaml.safe_load(f)
    return obj


def __find_roots(node, inputs, consider_load=True):
    """
    find the root nodes that the `node` dependes on

    Arguments
    -------
    node: Node
        the leaf node, of whom we need to find the dependent input nodes
    inputs: list
        resulting list to store all the root nodes in this list
    consider_load: bool
        whether it skips the node which are loading cache file or not
    Returns
    -----
    None

    """

    if (node.visited):
        return
    node.visited = True
    if len(node.inputs) == 0:
        inputs.append(node)
        return
    if consider_load and node.load:
        inputs.append(node)
        return
    for i in node.inputs:
        __find_roots(i, inputs, consider_load)


def get_graph(obj, replace):
    """
    compute the graph structure of the nodes. It will set the input and output
    nodes for each of the node

    Arguments
    -------
    obj: list
        a list of Python object that defines the nodes
    replace: dict
        conf parameters replacement
    Returns
    -----
    dict
        keys are Node unique ids
        values are Node objects

    """

    obj_dict = {}
    conf_dict = {}
    # instantiate objects
    for o in obj:
        if o['id'] in replace:
            o = copy.deepcopy(o)
            o.update(replace[o['id']])
        if isinstance(o['type'], str):
            if 'filepath' in o:
                spec = importlib.util.spec_from_file_location(o['id'],
                                                              o['filepath'])
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                NodeClass = getattr(mod, o['type'])
            else:
                NodeClass = getattr(mod_lib, o['type'])
        elif issubclass(o['type'], Node):
            NodeClass = o['type']
        else:
            raise "Not supported"
        load = False
        save = False
        if 'load' in o:
            load = o['load']
        if 'save' in o:
            save = o['save']
        instance = NodeClass(o['id'], o['conf'], load, save)
        obj_dict[o['id']] = instance
        conf_dict[o['id']] = o
    # build the graph
    for key in obj_dict:
        instance = obj_dict[key]
        for input_id in conf_dict[key]['inputs']:
            input_instance = obj_dict[input_id]
            instance.inputs.append(input_instance)
            input_instance.outputs.append(instance)

    # this part is to do static type checks
    raw_inputs = []
    for k in obj_dict.keys():
        __find_roots(obj_dict[k], raw_inputs, consider_load=False)
    for i in raw_inputs:
        i.columns_flow()
    # clean up the visited status for run computations
    for key in obj_dict:
        obj_dict[key].visited = False
    return obj_dict


def viz_graph(obj):
    """
    Generate the visulization of the graph in the JupyterLab

    Arguments
    -------
    obj: list
        a list of Python object that defines the nodes
    Returns
    -----
    nx.DiGraph
    """
    G = nx.DiGraph()
    # instantiate objects
    for o in obj:
        for i in o['inputs']:
            G.add_edge(i, o['id'])
    return G


def run(obj, outputs, replace={}):
    """
    Flow the dataframes in the graph to do the data science computations.

    Arguments
    -------
    obj: list
        a list of Python object that defines the nodes
    outputs: list
        a list of the leaf nodes that we need the final results
    replace: list
        a dict that defines the conf parameters replacement
    Returns
    -----
    tuple
        the results corresponding to the outputs list
    """

    obj_dict = get_graph(obj, replace)
    output_node = Node('unique_output', {})
    # want to save the intermediate results
    output_node.clear_input = False
    results = []
    results_obj = []
    for o in outputs:
        o_obj = obj_dict[o]
        results_obj.append(o_obj)
        output_node.inputs.append(o_obj)
        o_obj.outputs.append(output_node)

    inputs = []
    __find_roots(output_node, inputs, consider_load=True)
    # now clean up the graph, removed the node that is not used for computation
    for key in obj_dict:
        current_obj = obj_dict[key]
        if not current_obj.visited:
            for i in current_obj.inputs:
                i.outputs.remove(current_obj)
            current_obj.inputs = []
    for i in inputs:
        i.flow()
    for r_obj in results_obj:
        results.append(output_node.input_df[r_obj])
    # clean the results afterwards
    output_node.input_df = {}
    return tuple(results)
