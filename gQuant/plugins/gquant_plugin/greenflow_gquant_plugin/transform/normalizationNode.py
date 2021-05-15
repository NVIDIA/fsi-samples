from collections import OrderedDict
from greenflow.dataframe_flow import Node
from greenflow.dataframe_flow.portsSpecSchema import (ConfSchema,
                                                      PortsSpecSchema)
from greenflow.dataframe_flow.metaSpec import MetaData
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin
from ..node_hdf_cache import NodeHDFCacheMixin
from .data_obj import NormalizationData

__all__ = ['NormalizationNode']


class NormalizationNode(TemplateNodeMixin, NodeHDFCacheMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.INPUT_PORT_NAME = 'df_in'
        self.OUTPUT_PORT_NAME = 'df_out'
        self.INPUT_NORM_MODEL_NAME = 'norm_data_in'
        self.OUTPUT_NORM_MODEL_NAME = 'norm_data_out'
        port_type = PortsSpecSchema.port_type
        port_inports = {
            self.INPUT_PORT_NAME: {
                port_type: [
                    "pandas.DataFrame", "cudf.DataFrame",
                    "dask_cudf.DataFrame", "dask.dataframe.DataFrame"
                ]
            },
            self.INPUT_NORM_MODEL_NAME: {
                port_type: [
                    "greenflow_gquant_plugin.transform.data_obj.NormalizationData"  # noqa
                ]
            }
        }
        port_outports = {
            self.OUTPUT_PORT_NAME: {
                port_type: "${port:df_in}"
            },
            self.OUTPUT_NORM_MODEL_NAME: {
                port_type: [
                    "greenflow_gquant_plugin.transform.data_obj.NormalizationData"  # noqa
                ]
            },
        }
        self.template_ports_setup(
            in_ports=port_inports,
            out_ports=port_outports
        )

    def meta_setup(self):
        cols_required = {}
        required = {
            self.INPUT_PORT_NAME: cols_required,
            self.INPUT_NORM_MODEL_NAME: cols_required
        }
        if 'columns' in self.conf and self.conf.get('include', True):
            cols_required = {}
            for col in self.conf['columns']:
                cols_required[col] = None
            required = {
                self.INPUT_PORT_NAME: cols_required,
                self.INPUT_NORM_MODEL_NAME: cols_required
            }
        output_cols = {
            self.OUTPUT_PORT_NAME: required[self.INPUT_PORT_NAME],
            self.OUTPUT_NORM_MODEL_NAME: required[
                self.INPUT_NORM_MODEL_NAME]
        }
        input_meta = self.get_input_meta()
        if (self.INPUT_NORM_MODEL_NAME in input_meta and
                self.INPUT_PORT_NAME in input_meta):
            cols_required = input_meta[self.INPUT_NORM_MODEL_NAME]
            required = {
                self.INPUT_PORT_NAME: cols_required,
                self.INPUT_NORM_MODEL_NAME: cols_required
            }
            col_from_inport = input_meta[self.INPUT_PORT_NAME]
            output_cols = {
                self.OUTPUT_PORT_NAME: col_from_inport,
                self.OUTPUT_NORM_MODEL_NAME: cols_required
            }
        elif (self.INPUT_NORM_MODEL_NAME in input_meta and
              self.INPUT_PORT_NAME not in input_meta):
            cols_required = input_meta[self.INPUT_NORM_MODEL_NAME]
            required = {
                self.INPUT_PORT_NAME: cols_required,
                self.INPUT_NORM_MODEL_NAME: cols_required
            }
            output_cols = {
                self.OUTPUT_PORT_NAME: cols_required,
                self.OUTPUT_NORM_MODEL_NAME: cols_required
            }
        elif (self.INPUT_NORM_MODEL_NAME not in input_meta and
              self.INPUT_PORT_NAME in input_meta):
            col_from_inport = input_meta[self.INPUT_PORT_NAME]
            enums = [col for col in col_from_inport.keys()]
            if 'columns' in self.conf:
                if self.conf.get('include', True):
                    included_colums = self.conf['columns']
                else:
                    included_colums = [col for col in enums
                                       if col not in self.conf['columns']]
                cols_required = OrderedDict()
                for col in included_colums:
                    if col in col_from_inport:
                        cols_required[col] = col_from_inport[col]
                    else:
                        cols_required[col] = None
                required = {
                    self.INPUT_PORT_NAME: cols_required,
                    self.INPUT_NORM_MODEL_NAME: cols_required
                }
                output_cols = {
                    self.OUTPUT_PORT_NAME: col_from_inport,
                    self.OUTPUT_NORM_MODEL_NAME: cols_required
                }
        metadata = MetaData(inports=required, outports=output_cols)
        # The port INPUT_NORM_MODEL_NAME connection is optional. If not
        # connected do not set in required
        isconnected = \
            self.INPUT_NORM_MODEL_NAME in self.get_connected_inports()
        if not isconnected:
            metadata.inports.pop(self.INPUT_NORM_MODEL_NAME, None)
        return metadata

    def conf_schema(self):
        json = {
            "title": "Normalization Node configure",
            "type": "object",
            "description": "Normalize the columns to have zero mean and std 1",
            "properties": {
                "columns":  {
                    "type": "array",
                    "description": """an array of columns that need to
                     be normalized, or excluded from normalization depending
                     on the `incldue` flag state""",
                    "items": {
                        "type": "string"
                    }
                },
                "include":  {
                    "type": "boolean",
                    "description": """if set true, the `columns` need to be
                    normalized. if false, all dataframe columns except the
                    `columns` need to be normalized""",
                    "default": True
                },
            },
            "required": [],
        }
        ui = {}
        input_meta = self.get_input_meta()
        if self.INPUT_PORT_NAME in input_meta:
            col_from_inport = input_meta[self.INPUT_PORT_NAME]
            enums = [col for col in col_from_inport.keys()]
            json['properties']['columns']['items']['enum'] = enums
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        normalize the data to zero mean, std 1

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        input_df = inputs[self.INPUT_PORT_NAME]
        if self.INPUT_NORM_MODEL_NAME in inputs:
            norm_data = inputs[self.INPUT_NORM_MODEL_NAME].data
            input_meta = self.get_input_meta()
            means = norm_data['mean']
            stds = norm_data['std']
            col_from_inport = input_meta[self.INPUT_NORM_MODEL_NAME]
            cols = [i for i in col_from_inport.keys()]
            # cols.sort()
        else:
            # need to compute the mean and std
            if self.conf.get('include', True):
                cols = self.conf['columns']
            else:
                cols = input_df.columns.difference(
                    self.conf['columns']).values.tolist()
            # cols.sort()
            means = input_df[cols].mean()
            stds = input_df[cols].std()
        norm = (input_df[cols] - means) / stds
        col_dict = {i: norm[i] for i in cols}
        norm_df = input_df.assign(**col_dict)
        output = {}
        if self.outport_connected(self.OUTPUT_PORT_NAME):
            output.update({self.OUTPUT_PORT_NAME: norm_df})
        if self.outport_connected(self.OUTPUT_NORM_MODEL_NAME):
            norm_data = {"mean": means, "std": stds}
            payload = NormalizationData(norm_data)
            output.update({self.OUTPUT_NORM_MODEL_NAME: payload})
        return output
