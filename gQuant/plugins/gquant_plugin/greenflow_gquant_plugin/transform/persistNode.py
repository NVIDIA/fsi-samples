from greenflow.dataframe_flow import (ConfSchema, PortsSpecSchema, NodePorts,
                                      MetaData)
from greenflow_gquant_plugin._port_type_node import _PortTypesMixin
from greenflow.dataframe_flow import Node
from dask.dataframe import DataFrame as DaskDataFrame
from dask.distributed import wait
import dask.distributed


class PersistNode(_PortTypesMixin, Node):

    def init(self):
        _PortTypesMixin.init(self)

    def ports_setup(self):
        dy = PortsSpecSchema.dynamic
        port_type = PortsSpecSchema.port_type
        o_inports = {}
        o_outports = {}
        o_inports[self.INPUT_PORT_NAME] = {
            port_type: [DaskDataFrame],
            dy: True
        }
        input_connections = self.get_connected_inports()
        for port_name in input_connections.keys():
            if port_name != self.INPUT_PORT_NAME:
                determined_type = input_connections[port_name]
                o_outports[port_name] = {port_type: determined_type}
        return NodePorts(inports=o_inports, outports=o_outports)

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

    def meta_setup(self):
        input_meta = self.get_input_meta()
        input_connections = self.get_connected_inports()
        input_cols = {}
        output_cols = {}
        for port_name in input_connections.keys():
            if port_name in input_meta:
                meta = input_meta[port_name]
                input_cols[port_name] = meta
                output_cols[port_name] = meta
        meta_data = MetaData(inports=input_cols, outports=output_cols)
        return meta_data

    def process(self, inputs):
        # df = df.drop('datetime', axis=1)
        output = {}
        client = dask.distributed.client.default_client()
        input_connections = self.get_connected_inports()
        objs = []
        for port_name in input_connections.keys():
            if port_name != self.INPUT_PORT_NAME:
                df = inputs[port_name]
                objs.append(df)
        objs = client.persist(objs)
        wait([objs])
        for port_name in input_connections.keys():
            if port_name != self.INPUT_PORT_NAME:
                output[port_name] = objs.pop(0)
        return output
