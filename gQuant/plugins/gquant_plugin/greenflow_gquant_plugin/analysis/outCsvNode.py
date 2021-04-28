from greenflow.dataframe_flow import Node, PortsSpecSchema
import dask_cudf
from greenflow.dataframe_flow.util import get_file_path
from greenflow.dataframe_flow.portsSpecSchema import ConfSchema
from greenflow.dataframe_flow.metaSpec import MetaDataSchema
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin
from ..node_hdf_cache import NodeHDFCacheMixin


class OutCsvNode(TemplateNodeMixin, NodeHDFCacheMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.INPUT_PORT_NAME = 'df_in'
        self.OUTPUT_PORT_NAME = 'df_out'
        port_type = PortsSpecSchema.port_type
        port_inports = {
            self.INPUT_PORT_NAME: {
                port_type: [
                    "pandas.DataFrame", "cudf.DataFrame",
                    "dask_cudf.DataFrame", "dask.dataframe.DataFrame"
                ]
            },
        }
        port_outports = {
            self.OUTPUT_PORT_NAME: {
                port_type: "${port:df_in}"
            }
        }
        cols_required = {}
        addition = {}
        meta_inports = {
            self.INPUT_PORT_NAME: cols_required
        }
        meta_outports = {
            self.OUTPUT_PORT_NAME: {
                MetaDataSchema.META_OP: MetaDataSchema.META_OP_ADDITION,
                MetaDataSchema.META_REF_INPUT: self.INPUT_PORT_NAME,
                MetaDataSchema.META_DATA: addition
            }
        }
        self.template_ports_setup(
            in_ports=port_inports,
            out_ports=port_outports
        )
        self.template_meta_setup(
            in_ports=meta_inports,
            out_ports=meta_outports
        )

    def conf_schema(self):
        input_meta = self.get_input_meta()
        json = {
            "title": "Cvs output Configure",
            "type": "object",
            "description": """Dump the input datafram to the resulting csv file.
            the output filepath is defined as `path` in the `conf`.
            if only a subset of columns is needed for the csv file,
            enumerate the columns in the `columns` of the `conf`
            """,
            "properties": {
                "path":  {
                    "type": "string",
                    "description": """The output filepath for the csv"""
                },
                "columns": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": """array of columns to be selected for
                    the csv"""
                }
            },
            "required": ["path"],
        }
        ui = {}
        if self.INPUT_PORT_NAME in input_meta:
            col_from_inport = input_meta[self.INPUT_PORT_NAME]
            enums = [col for col in col_from_inport.keys()]
            json['properties']['columns']['items']['enum'] = enums
            return ConfSchema(json=json, ui=ui)
        else:
            return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        dump the input datafram to the resulting csv file.
        the output filepath is defined as `path` in the `conf`.
        if only a subset of columns is needed for the csv file, enumerate the
        columns in the `columns` of the `conf`

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """

        raw_input_df = inputs[self.INPUT_PORT_NAME]
        if 'columns' in self.conf:
            raw_input_df = raw_input_df[self.conf['columns']]
        if isinstance(raw_input_df,  dask_cudf.DataFrame):
            input_df = raw_input_df.compute()  # get the computed value
        else:
            input_df = raw_input_df
        input_df.to_pandas().to_csv(get_file_path(self.conf['path']),
                                    index=False)
        return {self.OUTPUT_PORT_NAME: raw_input_df}
