from greenflow.dataframe_flow import Node
import dask_cudf
from greenflow.dataframe_flow.util import get_file_path
from greenflow.dataframe_flow.portsSpecSchema import ConfSchema
from .._port_type_node import _PortTypesMixin


class OutCsvNode(Node, _PortTypesMixin):

    def init(self):
        _PortTypesMixin.init(self)
        self.INPUT_PORT_NAME = 'df_in'
        self.OUTPUT_PORT_NAME = 'df_out'

    def meta_setup(self):
        required = {}
        return _PortTypesMixin.meta_setup(self, required=required)

    def ports_setup(self):
        return _PortTypesMixin.ports_setup(self)

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
