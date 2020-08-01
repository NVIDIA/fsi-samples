from gquant.dataframe_flow import Node
from gquant.dataframe_flow import TaskGraph
from gquant.dataframe_flow.taskSpecSchema import TaskSpecSchema
from gquant.dataframe_flow.portsSpecSchema import ConfSchema
from gquant.dataframe_flow.portsSpecSchema import NodePorts

INPUT_ID = '4fd31358-fb80-4224-b35f-34402c6c3763'


class CompositeNode(Node):

    def ports_setup(self):
        required = {}
        inports = {}
        outports = {}
        if 'taskgraph' in self.conf:
            task_graph = TaskGraph.load_taskgraph(self.conf['taskgraph'])
            task_graph.build()
            if 'input' in self.conf and self.conf['input'] in task_graph:
                inputNode = task_graph[self.conf['input']]
                inputNode.inputs.clear()
                if hasattr(self, 'inputs'):
                    inputNode.inputs.extend(self.inputs)
                required = inputNode.required
                inports = inputNode.ports_setup().inports
            if 'output' in self.conf and self.conf['output'] in task_graph:
                outNode = task_graph[self.conf['output']]
                outports = outNode.ports_setup().outports
        self.required = required
        return NodePorts(inports=inports, outports=outports)

    def columns_setup(self):
        out_columns = {}
        if 'taskgraph' in self.conf:
            task_graph = TaskGraph.load_taskgraph(self.conf['taskgraph'])
            task_graph.build()
            if 'input' in self.conf and self.conf['input'] in task_graph:
                inputNode = task_graph[self.conf['input']]
                inputNode.inputs.clear()
                if hasattr(self, 'inputs'):
                    inputNode.inputs.extend(self.inputs)
            if 'output' in self.conf and self.conf['output'] in task_graph:
                outNode = task_graph[self.conf['output']]
                out_columns = outNode.columns_setup()
        return out_columns

    def conf_schema(self):
        json = {
            "title": "Composite Node configure",
            "type": "object",
            "description": """Use a sub taskgraph as a composite node""",
            "properties": {
                "taskgraph":  {
                    "type": "string",
                    "description": "the taskgraph filepath"
                },
                "input":  {
                    "type": "string",
                    "description": "the input node id"
                },
                "output":  {
                    "type": "string",
                    "description": "the output node id"
                }
            },
            "required": ["taskgraph", "input", "output"],
        }
        ui = {
            "taskgraph": {"ui:widget": "text"},
            "input": {"ui:widget": "text"},
            "output": {"ui:widget": "text"}
        }

        # input_columns = self.get_input_columns()
        # if (self.INPUT_PORT_LEFT_NAME in input_columns
        #         and self.INPUT_PORT_RIGHT_NAME in input_columns):
        #     col_left_inport = input_columns[self.INPUT_PORT_LEFT_NAME]
        #     col_right_inport = input_columns[self.INPUT_PORT_RIGHT_NAME]
        #     enums1 = set([col for col in col_left_inport.keys()])
        #     enums2 = set([col for col in col_right_inport.keys()])
        #     json['properties']['column']['enum'] = list(
        #         enums1.intersection(enums2))
        #     ui = {}
        #     return ConfSchema(json=json, ui=ui)
        # else:
        #     ui = {
        #         "column": {"ui:widget": "text"}
        #     }
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        Composite computation

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        if 'taskgraph' in self.conf:
            task_graph = TaskGraph.load_taskgraph(self.conf['taskgraph'])
            print('load', self.conf['taskgraph'])
            task_graph.build()
            if 'input' in self.conf and self.conf['input'] in task_graph:
                inputNode = task_graph[self.conf['input']]
                inputNode.inputs.clear()
                if hasattr(self, 'inputs'):
                    inputNode.inputs.extend(self.inputs)
                inports = inputNode.ports_setup().inports
            else:
                return {}
            if 'output' in self.conf and self.conf['output'] in task_graph:
                outNode = task_graph[self.conf['output']]
            else:
                return {}

            class InputFeed(Node):

                def columns_setup(self2):
                    output = {}
                    if hasattr(self, 'inputs'):
                        inputNode.inputs.extend(self.inputs)
                        for inp in self.inputs:
                            output[inp['to_port']] = inp['from_node'].columns_setup()[inp['from_port']]
                    # for key in inports.keys():
                    #     output[key] = {}
                    return output

                def ports_setup(self2):
                    return NodePorts(inports={}, outports=inports)

                def conf_schema(self2):
                    return ConfSchema()

                def process(self2, empty):
                    output = {}
                    for key in inports.keys():
                        output[key] = inputs[key]
                    return output

            obj = {
                TaskSpecSchema.task_id: INPUT_ID,
                TaskSpecSchema.conf: {},
                TaskSpecSchema.node_type: InputFeed,
                TaskSpecSchema.inputs: []
            }
            newInputs = {}
            for key in inports.keys():
                newInputs[key] = INPUT_ID+'.'+key
            task_graph.extend([obj])
            outputLists = []

            out_ports = outNode.ports_setup().outports
            for key in out_ports.keys():
                if self.outport_connected(key):
                    outputLists.append(outNode.uid+'.'+key)
            result = task_graph.run(outputLists, replace={inputNode.uid: {
                TaskSpecSchema.inputs: newInputs
            }})
            output = {}
            for key in result.get_keys():
                output[key.split('.')[1]] = result[key]
            return output
        else:
            return {}
