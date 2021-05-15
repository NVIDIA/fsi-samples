from .compositeNode import CompositeNode
from greenflow.dataframe_flow.portsSpecSchema import (ConfSchema,
                                                      PortsSpecSchema,
                                                      NodePorts)
from .data_obj import ConfData
from .json_util import parse_config
from jsonpath_ng import parse

__all__ = ["ContextCompositeNode"]


default_map = {
    "boolean": False,
    "number": 0.0,
    "string": "a string",
    "array": []
}


class ContextCompositeNode(CompositeNode):

    def init(self):
        super().init()
        self.INPUT_CONFIG = 'conf_in'
        self.OUTPUT_CONFIG = 'conf_out'

    def ports_setup(self):
        ports = super().ports_setup()
        port_type = PortsSpecSchema.port_type
        inports = ports.inports
        outports = ports.outports
        inports[self.INPUT_CONFIG] = {
            port_type: ConfData
        }
        outports[self.OUTPUT_CONFIG] = {
            port_type: ConfData
        }
        output_port = NodePorts(inports=inports, outports=outports)
        return output_port

    def meta_setup(self):
        out_meta = super().meta_setup()
        out_meta.outports[self.OUTPUT_CONFIG] = self.conf
        return out_meta

    def conf_schema(self):
        # import pdb
        # pdb.set_trace()
        task_graph = self.task_graph
        # cache_key, task_graph, replacementObj = self._compute_hash_key()
        # if cache_key in CACHE_SCHEMA:
        #     return CACHE_SCHEMA[cache_key]
        json = {
            "title": "Context Composite Node configure",
            "type": "object",
            "description": """Use a sub taskgraph as a composite node""",
            "properties": {
                "taskgraph":  {
                    "type": "string",
                    "description": "the taskgraph filepath"
                },
                "input":  {
                    "type": "array",
                    "description": "the input node ids",
                    "items": {
                        "type": "string"
                    }
                },
                "output":  {
                    "type": "array",
                    "description": "the output node ids",
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
            json['properties']['context'] = {
                    "type": "object",
                    "description": "context parameters",
                    "additionalProperties": {
                                "type": "object",
                                "description": """The context
                                parameters for this composite node""",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                    },
                                },
                                "dependencies": {
                                     "type": {
                                         "oneOf": []
                                     }
                                }
                            }
                }
            # import pdb
            # pdb.set_trace()
            all_fields = parse_config(self.replacementObj)
            types = list(all_fields.keys())
            addional = json['properties']['context']['additionalProperties']
            addional['properties']['type']['enum'] = types
            typelist = addional['dependencies']['type']['oneOf']

        for ty in types:
            ty_splits = ty.split('_')
            obj_temp = {
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "the parameter data type"
                    },
                    "value": {
                        "type": ty_splits[0],
                        "default": default_map[ty_splits[0]],
                        "description": "the value for this context parameter"
                    },
                    "map": {
                        "type": "array",
                        "description": """The fields of subnode's config this
                        parameter maps to""",
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
                }
            }
            if len(ty_splits) > 1:
                obj_temp['properties']['value']['items'] = {
                    "type": ty_splits[1]
                }
            type_container = all_fields[ty]
            ids = list(type_container.keys())
            obj_temp['properties']['type']['enum'] = [ty]
            obj_temp['properties']['map'][
                'items']['properties']['node_id']['enum'] = ids
            idlist = obj_temp['properties']['map'][
                'items']['dependencies']['node_id']['oneOf']
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
            typelist.append(obj_temp)

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
                        out_ports.append(node_id+'.'+port)
                    ids_in_graph.append(node_id)
            json['properties']['input']['items']['enum'] = in_ports
            json['properties']['output']['items']['enum'] = out_ports
        out_schema = ConfSchema(json=json, ui=ui)
        return out_schema

    def conf_update(self):
        input_conf = self.get_input_meta(self.INPUT_CONFIG)
        if input_conf is not None:
            self.conf.update(input_conf)

    def update_replace(self, replaceObj, task_graph, **kwargs):
        """
        called inside the self.process function to get the updated
        replacement object before taskgraph.run.
        """
        # find the other replacment conf
        if task_graph:
            for task in task_graph:
                key = task.get('id')
                newid = key
                conf = task.get('conf')
                if newid in replaceObj:
                    replaceObj[newid].update({'conf': conf})
                else:
                    replaceObj[newid] = {}
                    replaceObj[newid].update({'conf': conf})
        # replace the numbers from the context
        if 'context' in self.conf:
            for key in self.conf['context'].keys():
                val = self.conf['context'][key]['value']
                for map_obj in self.conf['context'][key]['map']:
                    xpath = map_obj['xpath']
                    expr = parse(xpath)
                    expr.update(replaceObj, val)

    def process(self, inputs):
        if self.INPUT_CONFIG in inputs:
            self.conf.update(inputs[self.INPUT_CONFIG].data)
        output = {}
        if self.outport_connected(self.OUTPUT_CONFIG):
            conf = ConfData(self.conf)
            output[self.OUTPUT_CONFIG] = conf
        more_output = super().process(inputs)
        output.update(more_output)
        return output
