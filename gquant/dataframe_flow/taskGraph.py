from collections import OrderedDict
import networkx as nx
import yaml
from .node import Node
from ._node_flow import OUTPUT_ID, OUTPUT_TYPE
from .task import Task
from .taskSpecSchema import TaskSpecSchema
from .portsSpecSchema import NodePorts, ConfSchema
import warnings
import copy


__all__ = ['TaskGraph', 'OutputCollector']


class NoDatesSafeLoader(yaml.SafeLoader):
    @classmethod
    def remove_implicit_resolver(cls, tag_to_remove):
        """
        Remove implicit resolvers for a particular tag

        Takes care not to modify resolvers in super classes.

        We want to load datetimes as strings, not dates, because we
        go on to serialise as json which doesn't have the advanced types
        of yaml, and leads to incompatibilities down the track.
        """
        if 'yaml_implicit_resolvers' not in cls.__dict__:
            cls.yaml_implicit_resolvers = cls.yaml_implicit_resolvers.copy()

        for first_letter, mappings in cls.yaml_implicit_resolvers.items():
            tag = [(tag, regexp) for tag,
                   regexp in mappings if tag != tag_to_remove]
            cls.yaml_implicit_resolvers[first_letter] = tag


NoDatesSafeLoader.remove_implicit_resolver('tag:yaml.org,2002:timestamp')


class OutputCollector(Node):
    def columns_setup(self):
        return super().columns_setup()

    def ports_setup(self):
        return NodePorts(inports={}, outports={})

    def conf_schema(self):
        return ConfSchema()

    def process(self, inputs):
        return super().process(inputs)


class Results(object):

    def __init__(self, values):
        self.values = tuple([i[1] for i in values])
        self.__keys = tuple([i[0] for i in values])
        self.__dict = OrderedDict(values)

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.values[key]
        else:
            return self.__dict[key]

    def __len__(self):
        return len(self.values)

    def __repr__(self):
        return "Results"+self.__dict.__repr__()[11:]

    def __str__(self):
        return "Results"+self.__dict.__str__()[11:]

    def __contains__(self, key):
        return True if key in self.__dict else False

    def get_keys(self):
        return self.__keys


def formated_result(result):
    import ipywidgets as widgets
    from IPython.display import display
    from ipywidgets import Output
    outputs = [Output() for i in range(len(result))]
    for i in range(len(result)):
        with outputs[i]:
            display(result[i])
    tab = widgets.Tab()
    tab.children = outputs
    for i in range(len(result)):
        tab.set_title(i, result.get_keys()[i])
    return tab


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
        self.__node_dict = {}
        self.__index = None
        # this is server widget that this taskgraph associated with
        self.__widget = None

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
            obj = yaml.load(f, Loader=NoDatesSafeLoader)
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
            if not isinstance(tod[TaskSpecSchema.node_type], str):
                tod[TaskSpecSchema.node_type] = tod[
                    TaskSpecSchema.node_type].__name__
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
            to_type = itask[TaskSpecSchema.node_type]
            if to_task == "":
                to_task = OUTPUT_TYPE
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

                if (to_type == OUTPUT_TYPE):
                    continue
                task_node = itask.get_node_obj()
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
        self.__node_dict.clear()
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
            nodetype = task[TaskSpecSchema.node_type]
            if (task_id == OUTPUT_ID or nodetype == OUTPUT_TYPE):
                output_task = Task({
                    TaskSpecSchema.task_id: OUTPUT_ID,
                    TaskSpecSchema.conf: {},
                    TaskSpecSchema.node_type: OutputCollector,
                    TaskSpecSchema.inputs: task[TaskSpecSchema.inputs]
                })
                node = output_task.get_node_obj(tgraph_mixin=True)
            else:
                node = task.get_node_obj(replace.get(task_id), profile,
                                         tgraph_mixin=True)
            self.__node_dict[task_id] = node

        # build the graph
        for task_id in self.__node_dict:
            node = self.__node_dict[task_id]
            task_inputs = node._task_obj[TaskSpecSchema.inputs]
            for input_idx, input_key in enumerate(task_inputs):
                # node_inputs should be a dict with entries:
                #     {iport: taskid.oport}
                input_task = task_inputs[input_key].split('.')
                dst_port = input_key

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
            i.validate_required_columns()

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

    def reset(self):
        self.__node_dict.clear()
        self.__task_list.clear()
        self.__index = None

    def _run(self, outputs=None, replace=None, profile=False, formated=False):
        replace = dict() if replace is None else replace

        self.build(replace, profile)

        graph_outputs = []
        # add the output graph only if the Output Node is not in the graph
        found_output_node = False
        for task in self:
            if (task[TaskSpecSchema.task_id] == OUTPUT_ID or
                    task[TaskSpecSchema.node_type] == OUTPUT_TYPE):
                found_output_node = True
                outputs_collector_node = self[task[TaskSpecSchema.task_id]]
                for input_item in outputs_collector_node.inputs:
                    from_node_id = input_item['from_node'].uid
                    fromStr = from_node_id+'.'+input_item['from_port']
                    graph_outputs.append(fromStr)
                break

        if outputs is None:
            outputs = graph_outputs

        if not found_output_node:
            output_task = Task({
                TaskSpecSchema.task_id: OUTPUT_ID,
                TaskSpecSchema.conf: {},
                TaskSpecSchema.node_type: OutputCollector,
                TaskSpecSchema.inputs: []
            })

            outputs_collector_node = output_task.get_node_obj(
                tgraph_mixin=True)

        outputs_collector_node.clear_input = False
        if not found_output_node or outputs is not None:
            if found_output_node:
                # clean all the connections to this output node
                for uid in self.__node_dict.keys():
                    node = self.__node_dict[uid]
                    node.outputs = list(filter(
                        lambda x: x['to_node'] != outputs_collector_node,
                        node.outputs))

                # remove the output
            # set the connection only if output_node is manullay created
            # or the output is overwritten
            outputs_collector_node.inputs.clear()
            outputs_collector_node.outputs.clear()
            for task_id in outputs:
                nodeid_oport = task_id.split('.')
                nodeid = nodeid_oport[0]
                oport = nodeid_oport[1] if len(nodeid_oport) > 1 else None
                onode = self.__node_dict[nodeid]
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

        results_task_ids = outputs

        inputs = []
        self.__find_roots(outputs_collector_node, inputs, consider_load=True)

        if self.__widget is not None:
            def progress_fun(uid):
                cacheCopy = copy.deepcopy(self.__widget.cache)
                nodes = list(filter(lambda x: x['id'] == uid,
                                    cacheCopy['nodes']
                                    if 'nodes' in cacheCopy else []))
                if len(nodes) > 0:
                    current_node = nodes[0]
                    current_node['busy'] = True
                self.__widget.cache = cacheCopy
            for i in inputs:
                i.flow(progress_fun)
            # clean up the progress

            def cleanup():
                import time
                cacheCopy = copy.deepcopy(self.__widget.cache)
                for node in cacheCopy.get('nodes', []):
                    node['busy'] = False
                time.sleep(1)
                self.__widget.cache = cacheCopy
            import threading
            t = threading.Thread(target=cleanup)
            t.start()
        else:
            for i in inputs:
                i.flow()

        results_dfs_dict = outputs_collector_node.input_df
        port_map = {}
        for input_item in outputs_collector_node.inputs:
            from_node_id = input_item['from_node'].uid
            fromStr = from_node_id+'.'+input_item['from_port']
            port_map[fromStr] = input_item['to_port']
        results = []
        for task_id in results_task_ids:
            results.append((task_id, results_dfs_dict[port_map[task_id]]))
        # clean the results afterwards
        outputs_collector_node.input_df = {}
        result = Results(results)
        if formated:
            return formated_result(result)
        else:
            return result

    def run(self, outputs=None, replace=None, profile=False, formated=False):
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
        if formated:
            import ipywidgets
            out = ipywidgets.Output(layout={'border': '1px solid black'})
            cap_run = out.capture(clear_output=True)(self._run)
            result = cap_run(outputs=outputs, replace=replace,
                             profile=profile, formated=formated)
            if result is None:
                result = ipywidgets.Tab()
            result.set_title(len(result.children), 'std output')
            result.children = result.children + (out,)
            return result
        else:
            return self._run(outputs=outputs, replace=replace, profile=profile,
                             formated=formated)

    def to_pydot(self, show_ports=False):
        nx_graph = self.viz_graph(show_ports=show_ports)
        to_pydot = nx.drawing.nx_pydot.to_pydot
        pdot = to_pydot(nx_graph)
        return pdot

    def get_widget(self):
        if self.__widget is None:
            from gquantlab.gquantmodel import GQuantWidget
            widget = GQuantWidget()
            widget.value = self.export_task_speclist()
            widget.set_taskgraph(self)
            self.__widget = widget
        return self.__widget

    def draw(self, show='lab', fmt='png', show_ports=False):
        if show in ('ipynb',):
            pdot = self.to_pydot(show_ports)
            pdot_out = pdot.create(format=fmt)
            if fmt in ('svg',):
                from IPython.display import SVG as Image  # @UnusedImport
            else:
                from IPython.display import Image  # @Reimport
            plt = Image(pdot_out)
            return plt
        else:
            widget = self.get_widget()
            return widget
