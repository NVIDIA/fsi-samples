from gquant.dataframe_flow.portsSpecSchema import (PortsSpecSchema,
                                                   NodePorts)
import cudf
import dask_cudf
import pandas as pd


class _PortTypesMixin(object):

    def init(self):
        self.INPUT_PORT_NAME = 'in'
        self.OUTPUT_PORT_NAME = 'out'

    def ports_setup_different_output_type(self, out_type):
        types = [cudf.DataFrame,
                 dask_cudf.DataFrame,
                 pd.DataFrame]
        port_type = PortsSpecSchema.port_type
        input_ports = {
            self.INPUT_PORT_NAME: {
                port_type: types
            }
        }
        output_ports = {
            self.OUTPUT_PORT_NAME: {
                port_type: out_type
            }
        }

        input_connections = self.get_connected_inports()
        if (self.INPUT_PORT_NAME in input_connections):
            determined_type = input_connections[self.INPUT_PORT_NAME]
            # connected
            return NodePorts(inports={self.INPUT_PORT_NAME: {
                port_type: determined_type}},
                outports=output_ports)
        else:
            return NodePorts(inports=input_ports, outports=output_ports)

    def ports_setup_from_types(self, types):
        port_type = PortsSpecSchema.port_type
        input_ports = {
            self.INPUT_PORT_NAME: {
                port_type: types
            }
        }

        output_ports = {
            self.OUTPUT_PORT_NAME: {
                port_type: types
            }
        }

        input_connections = self.get_connected_inports()
        if self.INPUT_PORT_NAME in input_connections:
            determined_type = input_connections[self.INPUT_PORT_NAME]
            # connected
            return NodePorts(inports={self.INPUT_PORT_NAME: {
                port_type: determined_type}},
                outports={self.OUTPUT_PORT_NAME: {
                    port_type: determined_type}})
        else:
            return NodePorts(inports=input_ports, outports=output_ports)

    def ports_setup(self):
        types = [cudf.DataFrame,
                 dask_cudf.DataFrame,
                 pd.DataFrame]
        return self.ports_setup_from_types(types)

    def columns_setup(self):
        input_columns = self.get_input_columns()
        if self.INPUT_PORT_NAME in input_columns:
            col_from_inport = input_columns[self.INPUT_PORT_NAME]
            output_cols = {
                self.OUTPUT_PORT_NAME: col_from_inport
            }
            return output_cols
        else:
            col_from_inport = self.required[self.INPUT_PORT_NAME]
            output_cols = {
                self.OUTPUT_PORT_NAME: col_from_inport
            }
            return output_cols

    def addition_columns_setup(self, addition):
        input_columns = self.get_input_columns()
        if self.INPUT_PORT_NAME not in input_columns:
            col_from_inport = self.required[self.INPUT_PORT_NAME]
        else:
            col_from_inport = input_columns[self.INPUT_PORT_NAME]
        # additional ports
        output_cols = {
            self.OUTPUT_PORT_NAME: addition
        }
        output_cols[self.OUTPUT_PORT_NAME].update(col_from_inport)
        return output_cols

    def deletion_columns_setup(self, deletion):
        input_columns = self.get_input_columns()
        if self.INPUT_PORT_NAME not in input_columns:
            col_from_inport = self.required[self.INPUT_PORT_NAME]
        else:
            col_from_inport = input_columns[self.INPUT_PORT_NAME]
        # delete the columns from the inputs
        for key in deletion:
            if key in col_from_inport:
                del col_from_inport[key]
        return {self.OUTPUT_PORT_NAME: col_from_inport}

    def retention_columns_setup(self, retention):
        return {self.OUTPUT_PORT_NAME: retention}
