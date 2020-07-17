from collections import OrderedDict
import networkx as nx
import yaml
from .node import Node
from ._node_flow import OUTPUT_ID
from .task import Task
from .taskSpecSchema import TaskSpecSchema
import warnings
import json



__all__ = ['TaskGraph']


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

    def __init__(self, task_spec_list=None):
        '''
        :param task_spec_list: List of task-spec dicts per TaskSpecSchema.
        '''
        self.__task_list = {}
        self.__index = None

        error_msg = 'Task-id "{}" already in the task graph. Set '\
                    'replace=True to replace existing task with extended task.'

        self.__extend(task_spec_list=task_spec_list, replace=False,
                      error_msg=error_msg)

    def __extend(self, task_spec_list=None, replace=False, error_msg=None):
        tspec_list = dict() if task_spec_list is None else task_spec_list

        if error_msg is None:
            error_msg = 'Task-id "{}" already in the task graph. Set '\
                        'replace=True to replace existing task.'

        for tspec in tspec_list:
            task = Task(tspec)
            task_id = task[TaskSpecSchema.task_id]
            if task_id in self.__task_list and not replace:
                raise Exception(error_msg.format(task_id))
            self.__task_list[task_id] = task

    def extend(self, task_spec_list=None, replace=False):
        '''
        Add more task-spec dicts to the graph

        :param task_spec_list: List of task-spec dicts per TaskSpecSchema.
        '''

        error_msg = 'Task-id "{}" already in the task graph. Set '\
                    'replace=True to replace existing task with extended task.'

        self.__extend(task_spec_list=task_spec_list, replace=replace,
                      error_msg=error_msg)

    def __contains__(self, task_id):
        return True if task_id in self.__task_list else False

    def __len__(self):
        return len(self.__task_list)

    def __iter__(self):
        self.__index = 0
        self.__tlist = list(self.__task_list.values())
        return self

    def __next__(self):
        idx = self.__index
        if idx is None or idx == len(self.__tlist):
            self.__index = None
            raise StopIteration
        task = self.__tlist[idx]
        self.__index = idx + 1
        return task

    def __find_roots(self, node, inputs, consider_load=True):
        """
        find the root nodes that the `node` depends on

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

        for node_in in node.inputs:
            inode = node_in['from_node']
            self.__find_roots(inode, inputs, consider_load)

    @staticmethod
    def load_taskgraph(filename):
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

    def export_task_speclist(self):
        tlist_od = []  # task list ordered
        for task in self:
            tod = OrderedDict([(TaskSpecSchema.task_id, 'idholder'),
                               (TaskSpecSchema.node_type, 'typeholder'),
                               (TaskSpecSchema.conf, 'confholder'),
                               (TaskSpecSchema.inputs, 'inputsholder')
                               ])
            tod.update(task._task_spec)
            tlist_od.append(tod)
        return tlist_od

    def save_taskgraph(self, filename):
        """
        Write a list of tasks i.e. taskgraph to a yaml file.

        Arguments
        -------
        filename: str
            The filename to write a yaml file to.

        """

        if not TaskGraph.__SETUP_YAML_ONCE:
            TaskGraph.setup_yaml()

        # we want -id to be first in the resulting yaml file.
        tlist_od = self.export_task_speclist()
        with open(filename, 'w') as fh:
            yaml.dump(tlist_od, fh, default_flow_style=False)

    def _repr_mimebundle_(self, include=None, exclude=None):
        obj = get_nodes(self)
        content = json.dumps(obj)
        return {'application/gquant-taskgraph': content}

    def viz_graph(self, show_ports=False):
        """
        Generate the visulization of the graph in the JupyterLab

        Returns
        -----
        nx.DiGraph
        """
        G = nx.DiGraph()
        # instantiate objects
        for itask in self:
            task_inputs = itask[TaskSpecSchema.inputs]
            to_task = itask[TaskSpecSchema.task_id]
            for iport_or_tid in task_inputs:
                # iport_or_tid: it is either to_port or task id (tid) b/c
                #     if using ports API task_inputs is a dictionary otherwise
                #     task_inputs is a list.
                taskin_and_oport = task_inputs[iport_or_tid] \
                    if isinstance(task_inputs, dict) else iport_or_tid
                isplit = taskin_and_oport.split('.')
                from_task = isplit[0]
                from_port = isplit[1] if len(isplit) > 1 else None
                if show_ports and from_port is not None:
                    to_port = iport_or_tid
                    common_tip = taskin_and_oport
                    G.add_edge(from_task, common_tip, label=from_port)
                    G.add_edge(common_tip, to_task, label=to_port)
                    tnode = G.nodes[common_tip]
                    tnode.update({
                        # 'label': '',
                        'shape': 'point'})
                else:
                    G.add_edge(from_task, to_task)

            # draw output ports
            if show_ports:
                task_node = itask.get_node_obj()
                if not task_node._using_ports():
                    continue
                # task_outputs = itask.get(TaskSpecSchema.outputs, [])
                for pout in task_node._get_output_ports():
                    out_tip = '{}.{}'.format(
                        itask[TaskSpecSchema.task_id], pout)
                    G.add_edge(to_task, out_tip, label=pout)
                    tnode = G.nodes[out_tip]
                    tnode.update({
                        # 'label': '',
                        'shape': 'point'})
        return G

    def build(self, replace=None, profile=False):
        """
        compute the graph structure of the nodes. It will set the input and
        output nodes for each of the node

        Arguments
        -------
        replace: dict
            conf parameters replacement
        """
        self.__node_dict = {}
        replace = dict() if replace is None else replace

        # check if there are item in the replace that is not in the graph
        task_ids = set([task[TaskSpecSchema.task_id] for task in self])
        for rkey in replace.keys():
            if rkey not in task_ids:
                warnings.warn(
                    'Replace task-id {} not found in task-graph'.format(rkey),
                    RuntimeWarning)

        # instantiate node objects
        for task in self:
            task_id = task[TaskSpecSchema.task_id]
            node = task.get_node_obj(replace.get(task_id), profile,
                                     tgraph_mixin=True)
            self.__node_dict[task_id] = node

        # build the graph
        for task_id in self.__node_dict:
            node = self.__node_dict[task_id]
            task_inputs = node._task_obj[TaskSpecSchema.inputs]
            for input_idx, input_key in enumerate(task_inputs):
                if node._using_ports():
                    # node_inputs should be a dict with entries:
                    #     {iport: taskid.oport}
                    input_task = task_inputs[input_key].split('.')
                    dst_port = input_key
                else:
                    input_task = input_key.split('.')
                    dst_port = input_idx

                input_id = input_task[0]
                src_port = input_task[1] if len(input_task) > 1 else None

                input_node = self.__node_dict[input_id]
                node.inputs.append({
                    'from_node': input_node,
                    'from_port': src_port,
                    'to_port': dst_port
                })
                # input_node.outputs.append(node)
                input_node.outputs.append({
                    'to_node': node,
                    'to_port': dst_port,
                    'from_port': src_port
                })

        # this part is to do static type checks
        raw_inputs = []
        for k in self.__node_dict.keys():
            self.__find_roots(self.__node_dict[k], raw_inputs,
                              consider_load=False)

        for i in raw_inputs:
            i.columns_flow()

        # clean up the visited status for run computations
        for task_id in self.__node_dict:
            self.__node_dict[task_id].visited = False

    def __getitem__(self, key):
        return self.__node_dict[key]

    def __str__(self):
        out_str = ""
        for k in self.__node_dict.keys():
            out_str += k + ": " + str(self.__node_dict[k]) + "\n"
        return out_str

    def run(self, outputs, replace=None, profile=False):
        """
        Flow the dataframes in the graph to do the data science computations.

        Arguments
        -------
        outputs: list
            a list of the leaf node IDs for which to return the final results
        replace: list
            a dict that defines the conf parameters replacement
        profile: Boolean
            whether profile the processing time of the nodes or not

        Returns
        -----
        tuple
            the results corresponding to the outputs list
        """
        replace = dict() if replace is None else replace

        self.build(replace, profile)

        class OutputCollector(Node):
            def columns_setup(self):
                super().columns_setup()

            def process(self, inputs):
                return super().process(inputs)

        output_task = Task({
            TaskSpecSchema.task_id: OUTPUT_ID,
            TaskSpecSchema.conf: {},
            TaskSpecSchema.node_type: OutputCollector,
            TaskSpecSchema.inputs: []
        })

        outputs_collector_node = output_task.get_node_obj(tgraph_mixin=True)

        # want to save the intermediate results
        outputs_collector_node.clear_input = False
        results = []
        results_task_ids = []
        for task_id in outputs:
            nodeid_oport = task_id.split('.')
            nodeid = nodeid_oport[0]
            oport = nodeid_oport[1] if len(nodeid_oport) > 1 else None
            onode = self.__node_dict[nodeid]
            results_task_ids.append(task_id)
            dummy_port = task_id
            outputs_collector_node.inputs.append({
                'from_node': onode,
                'from_port': oport,
                'to_port': dummy_port
            })
            onode.outputs.append({
                'to_node': outputs_collector_node,
                'to_port': dummy_port,
                'from_port': oport
            })

        inputs = []
        self.__find_roots(outputs_collector_node, inputs, consider_load=True)
        # now clean up the graph, removed the node that is not used for
        # computation
        for key in self.__node_dict:
            node_check_visit = self.__node_dict[key]
            if not node_check_visit.visited:
                for inode_info in node_check_visit.inputs:
                    inode = inode_info['from_node']
                    oport = inode_info['from_port']
                    iport = inode_info['to_port']
                    onode_info = {
                        'to_node': node_check_visit,
                        'to_port': iport,
                        'from_port': oport
                    }
                    inode.outputs.remove(onode_info)
                node_check_visit.inputs = []

        for i in inputs:
            i.flow()

        results_dfs_dict = outputs_collector_node.input_df
        for task_id in results_task_ids:
            results.append(results_dfs_dict[task_id])

        # clean the results afterwards
        outputs_collector_node.input_df = {}
        return tuple(results)

    def to_pydot(self, show_ports=False):
        nx_graph = self.viz_graph(show_ports=show_ports)
        to_pydot = nx.drawing.nx_pydot.to_pydot
        pdot = to_pydot(nx_graph)
        return pdot

    def draw(self, show='lab', fmt='png', show_ports=False):
        pdot = self.to_pydot(show_ports)
        pdot_out = pdot.create(format=fmt)

        if show in ('ipynb',):
            from IPython.display import display
            if fmt in ('svg',):
                from IPython.display import SVG as Image  # @UnusedImport
            else:
                from IPython.display import Image  # @Reimport

            plt = Image(pdot_out)
            display(plt)
        elif show in ('lab',):
            from gquantlab.gquantmodel import GQuantWidget
            widget = GQuantWidget()
            widget.value = self.export_task_speclist()
            widget.set_taskgraph(self)
            return widget
        else:
            return pdot_out
