import os
import importlib
import copy
from .taskSpecSchema import TaskSpecSchema
from ._node import _Node
from ._node_flow import OUTPUT_ID
from pathlib import Path
import sys
from collections import namedtuple
import configparser


class ConfigParser(configparser.ConfigParser):
    """Can get options() without defaults
    """

    def options(self, section, no_defaults=True, **kwargs):
        if no_defaults:
            try:
                return list(self._sections[section].keys())
            except KeyError:
                raise configparser.NoSectionError(section)
        else:
            return super().options(section, **kwargs)


def get_gquant_config_modules():
    if 'GQUANT_CONFIG' not in os.environ:
        os.environ['GQUANT_CONFIG'] = os.getcwd()+'/gquantrc'
        print(os.environ['GQUANT_CONFIG'])
    config = ConfigParser(defaults=os.environ)
    gquant_cfg = os.getenv('GQUANT_CONFIG', None)
    if Path(gquant_cfg).is_file():
        config.read(gquant_cfg)
    if 'ModuleFiles' not in config:
        return []
    modules_names = config.options('ModuleFiles')
    modules_list = {imod: config['ModuleFiles'][imod]
                    for imod in modules_names}
    return modules_list


__all__ = ['Task']

DEFAULT_MODULE = os.getenv('GQUANT_PLUGIN_MODULE', "gquant.plugin_nodes")


def load_modules(pathfile, name=None):
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
    if name is None:
        modulename = filename.stem
    else:
        modulename = name
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

    def set_output(self):
        """
        set the uniq output id to task
        """
        from .taskGraph import OutputCollector
        self._task_spec[TaskSpecSchema.task_id] = OUTPUT_ID
        self._task_spec[TaskSpecSchema.node_type] = OutputCollector

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
        module_name = task_spec.get(TaskSpecSchema.module)

        node_type = task_spec[TaskSpecSchema.node_type]
        task = Task(task_spec)

        # create a task to add path path
        def append_path(path):
            if path not in sys.path:
                sys.path.append(path)

        if isinstance(node_type, str):
            if modulepath is not None:
                loaded = load_modules(modulepath)
                module_dir = loaded.path
                mod = loaded.mod
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
                NodeClass = None
                try:
                    global DEFAULT_MODULE
                    plugmod = os.getenv('GQUANT_PLUGIN_MODULE', DEFAULT_MODULE)
                    MODLIB = importlib.import_module(plugmod)
                    NodeClass = getattr(MODLIB, node_type)
                except AttributeError:
                    modules = get_gquant_config_modules()
                    if (module_name is not None):
                        loaded = load_modules(
                            modules[module_name], name=module_name)
                        module_dir = loaded.path
                        mod = loaded.mod
                        try:
                            NodeClass = getattr(mod, node_type)
                        except AttributeError:
                            pass
                    else:
                        for key in modules:
                            loaded = load_modules(modules[key], name=key)
                            module_dir = loaded.path
                            mod = loaded.mod
                            try:
                                NodeClass = getattr(mod, node_type)
                                break
                            except AttributeError:
                                continue
                    if NodeClass is None:
                        raise Exception("Cannot find the Node Class:" +
                                        node_type)
                    else:
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
