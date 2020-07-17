from gquant.dataframe_flow import TaskGraph
from gquant.dataframe_flow import Node
from gquant.dataframe_flow.task import Task
from gquant.dataframe_flow.taskGraph import get_node_obj, get_nodes
import importlib
import pathlib
import gquant.plugin_nodes as plugin_nodes
import inspect
from gquant.dataframe_flow import Node
import uuid
from datetime import datetime as dt
import sys
sys.path.append('modules') # noqa E262
import os

print("Path at terminal when executing this file")
print(os.getcwd() + "\n")

print("This file path, relative to os.getcwd()")
print(__file__ + "\n")

print("This file full path (following symlinks)")
full_path = os.path.realpath(__file__)
print(full_path + "\n")


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
    print(all_modules)
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