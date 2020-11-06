import pandas as pd
import cudf
from dask.dataframe import DataFrame as DaskDataFrame
import dask_cudf

from gquant.dataframe_flow import Node
from gquant.dataframe_flow.portsSpecSchema import ConfSchema
from .._port_type_node import _PortTypesMixin

from gquant.dataframe_flow.portsSpecSchema import (
    PortsSpecSchema, NodePorts)

__all__ = ["DaskComputeNode"]


class DaskComputeNode(Node):

    def init(self):
        _PortTypesMixin.init(self)
        self.INPUT_PORT_NAME = 'in'
        self.OUTPUT_PORT_NAME = 'out'
        # no required columns
        self.required[self.INPUT_PORT_NAME] = {}

    def conf_schema(self):
        json = {
            "title": "Run dask compute",
            "type": "object",
            "description": "If the input is a dask or dask_cudf dataframe "
            "then run compute on it, otherwise pass through."
        }

        return ConfSchema(json=json)

    def ports_setup(self):
        port_type = PortsSpecSchema.port_type

        input_connections = self.get_connected_inports()
        if (self.INPUT_PORT_NAME in input_connections):
            determined_type = input_connections[self.INPUT_PORT_NAME]
            inports = {self.INPUT_PORT_NAME: {port_type: determined_type}}
        else:
            intypes = [cudf.DataFrame, dask_cudf.DataFrame,
                       pd.DataFrame, DaskDataFrame]
            inports = {self.INPUT_PORT_NAME: {port_type: intypes}}

        out_types = [cudf.DataFrame, pd.DataFrame]
        outports = {self.OUTPUT_PORT_NAME: {port_type: out_types}}

        return NodePorts(inports=inports, outports=outports)

    def meta_setup(self):
        '''Pass through columns from inputs to outputs'''
        return _PortTypesMixin.meta_setup(self)

    def process(self, inputs):
        din = inputs[self.INPUT_PORT_NAME]
        dout = din.compute() if isinstance(din, DaskDataFrame) else din

        return {self.OUTPUT_PORT_NAME: dout}
