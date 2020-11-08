from gquant.dataframe_flow.portsSpecSchema import (PortsSpecSchema,
                                                   MetaData,
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

    def meta_setup(self, required={}):
        input_meta = self.get_input_meta()
        if self.INPUT_PORT_NAME in input_meta:
            col_from_inport = input_meta[self.INPUT_PORT_NAME]
            output_cols = {
                self.OUTPUT_PORT_NAME: col_from_inport
            }
            input_cols = {
                self.INPUT_PORT_NAME: required
            }
            meta_data = MetaData(inports=input_cols, outports=output_cols)
            return meta_data
        else:
            input_cols = {
                self.INPUT_PORT_NAME: required
            }
            output_cols = {
                self.OUTPUT_PORT_NAME: required
            }
            meta_data = MetaData(inports=input_cols, outports=output_cols)
            return meta_data

    def addition_meta_setup(self, addition, required={}):
        input_meta = self.get_input_meta()
        if self.INPUT_PORT_NAME not in input_meta:
            col_from_inport = required
        else:
            col_from_inport = input_meta[self.INPUT_PORT_NAME]
        input_cols = {
            self.INPUT_PORT_NAME: required
        }
        # additional ports
        output_cols = {
            self.OUTPUT_PORT_NAME: addition
        }
        output_cols[self.OUTPUT_PORT_NAME].update(col_from_inport)
        meta_data = MetaData(inports=input_cols, outports=output_cols)
        return meta_data

    def deletion_meta_setup(self, deletion, required={}):
        input_meta = self.get_input_meta()
        if self.INPUT_PORT_NAME not in input_meta:
            col_from_inport = required
        else:
            col_from_inport = input_meta[self.INPUT_PORT_NAME]
        # delete the columns from the inputs
        input_cols = {
            self.INPUT_PORT_NAME: required
        }
        for key in deletion:
            if key in col_from_inport:
                del col_from_inport[key]
        meta_data = MetaData(inports=input_cols,
                             outports={self.OUTPUT_PORT_NAME: col_from_inport})
        return meta_data

    def retention_meta_setup(self, retention, required={}):
        input_cols = {
            self.INPUT_PORT_NAME: required
        }
        meta_data = MetaData(inports=input_cols,
                             outports={self.OUTPUT_PORT_NAME: retention})
        return meta_data
