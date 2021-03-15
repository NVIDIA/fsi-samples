from greenflow.plugin_nodes import CompositeNode
from greenflow.plugin_nodes import ContextCompositeNode
from greenflow.dataframe_flow.portsSpecSchema import (ConfSchema,
                                                      PortsSpecSchema,
                                                      NodePorts)
from greenflow.plugin_nodes.util.json_util import parse_config
from dask.dataframe import DataFrame as DaskDataFrame
import dask
from jsonpath_ng import parse
import dask.distributed

__all__ = ["SimpleParallelNode"]


class SimpleParallelNode(CompositeNode):
    """
    A SimpleParallelNode that can take a single GPU/CPU workflow, and convert
    it to output Dask dataframe. Each partition in the Dask dataframe will be
    computed in parallel in different GPU/CPUs.

    In the SimpleParallelNode configuration, user just need to set the
    taskgraph, the inputs and outputs of the taskgraph,
    the context parameters for the taskgraph. Most importantly,
    it can map the iteration id (or the Dask Dataframe partition id) to any
     number typed configuration item of the taskgraph.
     E.g. it can be used to set the randomn seed number for each of the
     iteration runs.
    """

    def init(self):
        super().init()

    def ports_setup(self):
        ports = super().ports_setup()
        port_type = PortsSpecSchema.port_type
        inports = ports.inports
        outports = ports.outports
        for k in outports.keys():
            outports[k][port_type] = [DaskDataFrame]
        output_port = NodePorts(inports=inports, outports=outports)
        return output_port

    def meta_setup(self):
        out_meta = super().meta_setup()
        return out_meta

    def conf_schema(self):
        task_graph = self.task_graph
        replacementObj = self.replacementObj
        # cache_key, task_graph, replacementObj = self._compute_has
        # cache_key, task_graph, replacementObj = self._compute_hash_key()
        # if cache_key in CACHE_SCHEMA:
        #     return CACHE_SCHEMA[cache_key]
        conf = ContextCompositeNode.conf_schema(self)
        json = {
            "title": "Simple Parallel Node",
            "type": "object",
            "description": """The SimpleParallelNode is used to parallelize the
             embarrassingly parallelizable cudf computation taskgraph.""",
            "properties": {
                "taskgraph": {
                    "type": "string",
                    "description": "the taskgraph filepath"
                },
                "input": {
                    "type": "array",
                    "description": "the input node ids",
                    "items": {
                        "type": "string"
                    }
                },
                "output": {
                    "type": "array",
                    "description": "the output node ids",
                    "items": {
                        "type": "string"
                    },
                },
                "iterations": {
                    "type": "integer",
                    "description": "the number of iterations",
                    "items": {
                        "type": "string"
                    }
                },
            },
            "required": ["taskgraph"]
        }
        ui = {
            "taskgraph": {"ui:widget": "TaskgraphSelector"},
        }

        types = []
        if 'taskgraph' in self.conf:
            if 'context' in conf.json['properties']:
                json['properties']['context'] = conf.json['properties'][
                    'context']
            json['properties']['map'] = {
                "type": "array",
                "description": """The iteration number maps to""",
                "items": {
                    "type": "object",
                    "properties": {
                        "node_id": {
                            "type": "string",
                            "enum": []
                        }
                    },
                    "dependencies": {
                        "node_id": {
                            "oneOf": [],
                        }
                    }
                }
            }
            all_fields = parse_config(replacementObj)
            types = list(all_fields.keys())
        if 'number' in types:
            ty = 'number'
            type_container = all_fields[ty]
            ids = list(type_container.keys())
            json['properties']['map']['items']['properties']['node_id'][
                'enum'] = ids
            idlist = json['properties']['map']['items']['dependencies'][
                'node_id']['oneOf']
            for subid in ids:
                id_obj = {
                    "properties": {
                        "node_id": {
                            "type": "string"
                        },
                        "xpath": {
                            "type": "string",
                        }
                    }
                }
                content = type_container[subid]
                paths = [i['path'] for i in content]
                names = [i['item'] for i in content]
                id_obj['properties']['node_id']['enum'] = [subid]
                id_obj['properties']['xpath']['enum'] = paths
                id_obj['properties']['xpath']['enumNames'] = names
                idlist.append(id_obj)

        pandas_df_name = 'pandas.core.frame.DataFrame'
        cudf_df_name = 'cudf.core.dataframe.DataFrame'

        if 'taskgraph' in self.conf:

            def inputNode_fun(inputNode, in_ports):
                pass

            def outNode_fun(outNode, out_ports):
                pass

            self._make_sub_graph_connection(task_graph,
                                            inputNode_fun, outNode_fun)

            ids_in_graph = []
            in_ports = []
            out_ports = []
            for t in task_graph:
                node_id = t.get('id')
                if node_id != '':
                    node = task_graph[node_id]
                    all_ports = node.ports_setup()
                    for port in all_ports.inports.keys():
                        in_ports.append(node_id+'.'+port)
                    for port in all_ports.outports.keys():
                        types = all_ports.outports[port][
                            PortsSpecSchema.port_type]
                        correct_type = False
                        if isinstance(types, list):
                            t_names = [
                                t.__module__ + '.' + t.__name__ for t in types
                            ]
                            if (pandas_df_name in t_names
                                    or cudf_df_name in t_names):
                                correct_type = True
                        else:
                            t_names = types.__module__ + '.' + types.__name__
                            if (pandas_df_name == t_names
                                    or cudf_df_name == t_names):
                                correct_type = True
                        if correct_type:
                            out_ports.append(node_id+'.'+port)
                    ids_in_graph.append(node_id)
            json['properties']['input']['items']['enum'] = in_ports
            json['properties']['output']['items']['enum'] = out_ports
        out_schema = ConfSchema(json=json, ui=ui)
        # CACHE_SCHEMA[cache_key] = out_schema
        return out_schema

    def conf_update(self):
        """
        run after init, used to update configuration
        """
        pass

    def update_replace(self, replaceObj, task_graph, **kwargs):
        """
        this method is called in the update

        @para replaceObj is a dictionary of the configuration
        @para task_graph is the task_graph loaded for this composite node
        @iternum integer, the iteration number

        It is intented to construct a new python dictionary to pass to the
        task_graph.run replace argument. So the composite taskgraph can run
         with different configurations.
        """
        ContextCompositeNode.update_replace(self, replaceObj, task_graph,
                                            **kwargs)
        # replace the numbers from the context
        if 'map' in self.conf and 'iternum' in kwargs:
            for i in range(len(self.conf['map'])):
                val = kwargs['iternum']
                map_obj = self.conf['map'][i]
                xpath = map_obj['xpath']
                expr = parse(xpath)
                expr.update(replaceObj, val)

    def process(self, inputs, **kwargs):
        output = {}
        # more_output = self._process(inputs)
        # output.update(more_output)
        iterations = self.conf['iterations']
        out_dfs = [
            dask.delayed(CompositeNode.process)(self, inputs, iternum=i)
            for i in range(iterations)
        ]
        client = dask.distributed.client.default_client()
        out_dfs = client.persist(out_dfs)
        meta = self.meta_setup().outports
        ports = self.ports_setup()
        for name in ports.outports.keys():
            if self.outport_connected(name):
                meta_data = meta[name]
                objs = [i[name] for i in out_dfs]
                dask_df = dask.dataframe.from_delayed(objs, meta=meta_data)
                output[name] = dask_df
        return output
