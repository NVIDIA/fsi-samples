from .node import Node
from collections import OrderedDict
import importlib
import os
import networkx as nx
import yaml


DEFAULT_MODULE = os.getenv('GQUANT_PLUGIN_MODULE', "gquant.plugin_nodes")
mod_lib = importlib.import_module(DEFAULT_MODULE)

__all__ = ['TaskGraph']


class Task(object):
    ''' A strong typed Task class that is converted from dictionary.
    '''

    def __typecheck(self, key):
        if (key == 'id'):
            assert isinstance(self.id, str)
        elif key == 'type':
            assert (isinstance(self.type, str) or issubclass(self.type, Node))
        elif key == 'conf':
            assert (isinstance(self.conf, dict) or isinstance(self.conf, list))
        elif key == 'filepath':
            assert isinstance(self.filepath, str)
        elif key == 'inputs':
            assert isinstance(self.inputs, list)
            for item in self.inputs:
                assert isinstance(item, str)
        elif key == 'load':
            assert isinstance(self.load, bool)
        elif key == 'save':
            assert isinstance(self.save, bool)
        else:
            if key in self.__dict__:
                del self.__dict__[key]
            raise KeyError

    def __init__(self, obj):
        self.id = obj['id']
        self.__typecheck('id')
        self.type = obj['type']
        self.__typecheck('type')
        self.conf = obj['conf']
        self.__typecheck('conf')
        if 'filepath' in obj:
            self.filepath = obj['filepath']
            self.__typecheck('conf')
        if 'load' in obj:
            self.load = obj['load']
            self.__typecheck('load')
        if 'save' in obj:
            self.save = obj['save']
            self.__typecheck('save')
        self.inputs = obj['inputs']
        self.__typecheck('inputs')

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value
        self.__typecheck(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def getNodeObj(self, replace={}):
        """
        instantiate a node instance for this task given the replacement setup

        Arguments
        -------
        replace: dict
            conf parameters replacement

        Returns
        -----
        object
            Node instance
        """
        objId = replace.get('id', self['id'])
        objPath = replace.get('filepath', self.get('filepath'))

        objType = replace.get('type', self['type'])
        objConf = replace.get('conf', self['conf'])

        load = replace.get('load', self.get('load', False))
        save = replace.get('save', self.get('save', False))

        if isinstance(objType, str):
            if objPath is not None:
                spec = importlib.util.spec_from_file_location(objId,
                                                              objPath)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                NodeClass = getattr(mod, objType)
            else:
                NodeClass = getattr(mod_lib, objType)
        elif issubclass(objType, Node):
            NodeClass = objType
        else:
            raise "Not supported"

        instance = NodeClass(objId, objConf, load, save)
        return instance


class TaskGraph(object):
    ''' TaskGraph class that is used to store the graph.
    '''

    __SETUP_YAML_ONCE = False

    @staticmethod
    def setup_yaml():
        '''Write out yaml in order for OrderedDict.'''
        # https://stackoverflow.com/a/8661021
        # https://stackoverflow.com/questions/47692094/lambda-works-
        # defined-function-does-not  # noqa
        # represent_dict_order = lambda dumper, data: \
        #     dumper.represent_mapping('tag:yaml.org,2002:map', data.items())
        def represent_dict_order(dumper, data):
            return dumper.represent_mapping('tag:yaml.org,2002:map',
                                            data.items())
        yaml.add_representer(OrderedDict, represent_dict_order)

        TaskGraph.__SETUP_YAML_ONCE = True

    def __init__(self, objs=None):
        self.__task_list = []
        self.__index = 0
        if objs is not None:
            for obj in objs:
                self.__task_list.append(Task(obj))

    def __len__(self):
        return len(self.__task_list)

    def __iter__(self):
        self.__index = 0
        return self

    def __next__(self):
        if self.__index == len(self.__task_list):
            raise StopIteration
        obj = self.__task_list[self.__index]
        self.__index += 1
        return obj

    def __find_roots(self, node, inputs, consider_load=True):
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
            self.__find_roots(i, inputs, consider_load)

    @staticmethod
    def load(objs):
        """
        load Python objects (list of dicts) to TaskGraph object

        Arguments
        -------
        objs: list
            the list of Python objects
        Returns
        -----
        object
            the TaskGraph instance

        """

        t = TaskGraph(objs)
        return t

    @staticmethod
    def load_workflow(filename):
        """
        load the yaml file to TaskGraph object

        Arguments
        -------
        filename: str
            the filename pointing to the yaml file in the filesystem
        Returns
        -----
        object
            the TaskGraph instance

        """

        with open(filename) as f:
            obj = yaml.safe_load(f)
        t = TaskGraph(obj)
        return t

    def save_workflow(self, filename):
        """
        Write a list of tasks i.e. workflow to a yaml file.

        Arguments
        -------
        filename: str
            The filename to write a yaml file to.

        """

        if not TaskGraph.__SETUP_YAML_ONCE:
            TaskGraph.setup_yaml()

        # we want -id to be first in the resulting yaml file.
        tlist_od = []  # task list ordered
        for task in self.__task_list:
            tod = OrderedDict([('id', 'idholder'),
                               ('type', 'typeholder'),
                               ('conf', 'confholder'),
                               ('inputs', 'inputsholder')
                               ])
            tod.update(task.__dict__)
            tlist_od.append(tod)

        with open(filename, 'w') as fh:
            yaml.dump(tlist_od, fh, default_flow_style=False)

    def viz_graph(self):
        """
        Generate the visulization of the graph in the JupyterLab

        Returns
        -----
        nx.DiGraph
        """
        G = nx.DiGraph()
        # instantiate objects
        for o in self.__task_list:
            for i in o['inputs']:
                G.add_edge(i, o['id'])
        return G

    def build(self, replace=None):
        """
        compute the graph structure of the nodes. It will set the input and
        output nodes for each of the node

        Arguments
        -------
        replace: dict
            conf parameters replacement
        """
        self.__task_dict = {}
        replace = dict() if replace is None else replace
        task_spec_dict = {}
        # instantiate objects
        for task_spec in self.__task_list:
            instance = task_spec.getNodeObj(replace.get(task_spec['id'], {}))
            self.__task_dict[task_spec['id']] = instance
            task_spec_dict[task_spec['id']] = task_spec

        # build the graph
        for task_id in self.__task_dict:
            instance = self.__task_dict[task_id]
            for input_id in task_spec_dict[task_id]['inputs']:
                input_instance = self.__task_dict[input_id]
                instance.inputs.append(input_instance)
                input_instance.outputs.append(instance)

        # this part is to do static type checks
        raw_inputs = []
        for k in self.__task_dict.keys():
            self.__find_roots(self.__task_dict[k], raw_inputs,
                              consider_load=False)

        for i in raw_inputs:
            i.columns_flow()

        # clean up the visited status for run computations
        for task_id in self.__task_dict:
            self.__task_dict[task_id].visited = False

    def __getitem__(self, key):
        return self.__task_dict[key]

    def __str__(self):
        out_str = ""
        for k in self.__task_dict.keys():
            out_str += k + ": " + str(self.__task_dict[k]) + "\n"
        return out_str

    def run(self, outputs, replace=None):
        """
        Flow the dataframes in the graph to do the data science computations.

        Arguments
        -------
        outputs: list
            a list of the leaf node IDs for which to return the final results
        replace: list
            a dict that defines the conf parameters replacement

        Returns
        -----
        tuple
            the results corresponding to the outputs list
        """
        replace = dict() if replace is None else replace
        self.build(replace)
        output_node = Node('unique_output', {})
        # want to save the intermediate results
        output_node.clear_input = False
        results = []
        results_obj = []
        for o in outputs:
            o_obj = self.__task_dict[o]
            results_obj.append(o_obj)
            output_node.inputs.append(o_obj)
            o_obj.outputs.append(output_node)

        inputs = []
        self.__find_roots(output_node, inputs, consider_load=True)
        # now clean up the graph, removed the node that is not used for
        # computation
        for key in self.__task_dict:
            current_obj = self.__task_dict[key]
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


if __name__ == "__main__":
    t = {'id': 'test',
         'type': "DropNode",
         'conf': {},
         'inputs': ["node_other"]}
    task = Task(t)
