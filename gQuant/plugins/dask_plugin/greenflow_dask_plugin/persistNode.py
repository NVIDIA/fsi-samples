from greenflow.dataframe_flow import (ConfSchema, PortsSpecSchema)
from greenflow.dataframe_flow.simpleNodeMixin import SimpleNodeMixin
from greenflow.dataframe_flow import Node
from dask.dataframe import DataFrame as DaskDataFrame
import dask.distributed


class PersistNode(SimpleNodeMixin, Node):

    def init(self):
        SimpleNodeMixin.init(self)
        port_type = PortsSpecSchema.port_type
        dy = PortsSpecSchema.dynamic
        self.INPUT_PORT_NAME = 'in'
        self.port_inports = {
            self.INPUT_PORT_NAME: {
                port_type: [
                    "dask_cudf.DataFrame", "dask.dataframe.DataFrame",
                    "builtins.object"
                ],
                dy: {
                    self.DYN_MATCH: True
                }
            },
        }

    def ports_setup(self):
        return SimpleNodeMixin.ports_setup(self)

    def meta_setup(self):
        return SimpleNodeMixin.meta_setup(self)

    def conf_schema(self):
        json = {
            "title": "Persist the dask dataframe",
            "type": "object",
            "properties": {
            },
        }

        ui = {
        }
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        # df = df.drop('datetime', axis=1)
        input_connections = self.get_connected_inports()
        determined_type = None
        for port_name in input_connections.keys():
            if port_name != self.INPUT_PORT_NAME:
                determined_type = input_connections[port_name]

        output = {}
        if (determined_type[0] is not None and issubclass(determined_type[0],
                                                          DaskDataFrame)):
            client = dask.distributed.client.default_client()
            input_connections = self.get_connected_inports()
            objs = []
            for port_name in input_connections.keys():
                if port_name != self.INPUT_PORT_NAME:
                    df = inputs[port_name]
                    objs.append(df)
            objs = client.persist(objs)
            for port_name in input_connections.keys():
                if port_name != self.INPUT_PORT_NAME:
                    output[port_name] = objs.pop(0)
        else:
            for port_name in input_connections.keys():
                if port_name != self.INPUT_PORT_NAME:
                    df = inputs[port_name]
                    output[port_name] = df
        return output
