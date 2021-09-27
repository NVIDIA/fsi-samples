from .node import Node
from .portsSpecSchema import NodePorts, ConfSchema


__all__ = ['Output_Collector', 'OUTPUT_ID', 'OUTPUT_TYPE']


class Output_Collector(Node):
    def meta_setup(self):
        return super().meta_setup()

    def ports_setup(self):
        return NodePorts(inports={}, outports={})

    def conf_schema(self):
        return ConfSchema()

    def process(self, inputs):
        return super().process(inputs)


# TODO: DO NOT RELY ON special OUTPUT_ID.
# OUTPUT_ID = 'f291b900-bd19-11e9-aca3-a81e84f29b0f_uni_output'
OUTPUT_ID = 'collector_id_fd9567b6'
OUTPUT_TYPE = Output_Collector.__name__
