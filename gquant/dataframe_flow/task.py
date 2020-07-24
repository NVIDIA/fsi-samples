import os
import importlib
import copy
from .taskSpecSchema import TaskSpecSchema
from ._node import _Node
from pathlib import Path
import sys
from python_settings import settings
from collections import namedtuple


__all__ = ['Task']

DEFAULT_MODULE = os.getenv('GQUANT_PLUGIN_MODULE', "gquant.plugin_nodes")


def load_modules_from_file(modulefile):
    """
    Given a py filename without path information,
    this method will find it from a set of paths from settings.MODULE_PATHS
    check for https://github.com/charlsagente/python-settings to
    learn how to set up the setting file. It will load the file as a python
    module, put it into the sys.modules and add the path into the pythonpath.
    @param modulefile
        string, file name
    @returns
        namedtuple, absolute path and loaded module
    """
    modulepaths = settings.MODULE_PATH
    found = False
    for path in modulepaths:
        filename = Path(path+'/'+modulefile)
        if filename.exists():
            found = True
            break
    if (not found):
        raise ("cannot find file %s" % (modulefile))
    return load_modules(str(filename))


def load_modules(pathfile):
    """
    Given a py filename with path information,
    It will load the file as a python
    module, put it into the sys.modules and add the path into the pythonpath.
    @param modulefile
        string, file name
    @returns
        namedtuple, absolute path and loaded module
    """
    filename = Path(pathfile)
    modulename = filename.stem
    spec = importlib.util.spec_from_file_location(modulename, str(filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    module_dir = str(filename.parent.absolute())
    spec.loader.exec_module(mod)
    Load = namedtuple("Load", "path mod")
    return Load(module_dir, mod)


class Task(object):
    ''' A strong typed Task class that is converted from dictionary.
    '''

    def __init__(self, task_spec):

        self._task_spec = {}  # internal dict

        # whatever is passed in has to be valid
        TaskSpecSchema.validate(task_spec)
        self._task_spec = copy.copy(task_spec)
        # deepcopies of inputs can still be done
        self._task_spec[TaskSpecSchema.inputs] = \
            copy.deepcopy(task_spec[TaskSpecSchema.inputs])

    def __getitem__(self, key):
        return self._task_spec[key]

    def get(self, key, default=None):
        return self._task_spec.get(key, default)

    def get_node_obj(self, replace=None, profile=False, tgraph_mixin=False):
        """
        instantiate a node instance for this task given the replacement setup

        Arguments
        -------
        replace: dict
            conf parameters replacement
        profile: Boolean
            profile the node computation

        Returns
        -----
        object
            Node instance
        """
        replace = dict() if replace is None else replace

        task_spec = copy.copy(self._task_spec)
        task_spec.update(replace)

        # node_id = task_spec[TaskSpecSchema.task_id]
        modulepath = task_spec.get(TaskSpecSchema.filepath)

        node_type = task_spec[TaskSpecSchema.node_type]
        task = Task(task_spec)

        if isinstance(node_type, str):
            if modulepath is not None:
                loaded = load_modules_from_file(modulepath)
                module_dir = loaded.path
                mod = loaded.mod

                # create a task to add path path
                def append_path(path):
                    if path not in sys.path:
                        sys.path.append(path)

                append_path(module_dir)

                try:
                    # add python path to all the client workers
                    # assume all the worikers share the same directory
                    # structure
                    import dask.distributed
                    client = dask.distributed.client.default_client()
                    client.run(append_path, module_dir)
                except (ValueError, ImportError):
                    pass
                NodeClass = getattr(mod, node_type)
            else:
                global DEFAULT_MODULE
                plugmod = os.getenv('GQUANT_PLUGIN_MODULE', DEFAULT_MODULE)
                # MODLIB = importlib.import_module(DEFAULT_MODULE)
                MODLIB = importlib.import_module(plugmod)
                NodeClass = getattr(MODLIB, node_type)
        elif issubclass(node_type, _Node):
            NodeClass = node_type
        else:
            raise Exception("Node type not supported: {}".format(node_type))

        assert issubclass(NodeClass, _Node), \
            'Node-type is not a subclass of "Node" class.'

        if tgraph_mixin:
            from ._node_flow import NodeTaskGraphMixin

            class NodeInTaskGraph(NodeTaskGraphMixin, NodeClass):
                def __init__(self, task):
                    NodeClass.__init__(self, task)
                    NodeTaskGraphMixin.__init__(self)

                def __repr__(self):
                    '''Override repr to show the name and path of the plugin
                    node class.'''
                    return '<{} {}.{} object at {}>'.format(
                        self.__class__.__name__,
                        NodeClass.__module__,
                        NodeClass.__name__,
                        hex(id(self)))

            node = NodeInTaskGraph(task)
        else:
            node = NodeClass(task)
        node.profile = profile
        return node


if __name__ == "__main__":
    t = {'id': 'test',
         'type': "DropNode",
         'conf': {},
         'inputs': ["node_other"]}
    task = Task(t)
