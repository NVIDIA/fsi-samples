from greenflow.dataframe_flow import Node, PortsSpecSchema
from greenflow.dataframe_flow.portsSpecSchema import ConfSchema
from greenflow.dataframe_flow.metaSpec import MetaDataSchema
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin
from ..node_hdf_cache import NodeHDFCacheMixin

__all__ = ["OneHotEncodingNode"]


class OneHotEncodingNode(TemplateNodeMixin, NodeHDFCacheMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.INPUT_PORT_NAME = 'in'
        self.OUTPUT_PORT_NAME = 'out'
        self.delayed_process = True
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
                port_type: "${port:in}"
            }
        }
        cols_required = {}
        addition = {}
        for col in self.conf:
            for cat in col['cats']:
                name = col.get('prefix')+col.get('prefix_sep', '_')+str(cat)
                addition.update({name: col.get('dtype', 'float64')})
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
        json = {
            "title": "One Hot Encoding configure",
            "type": "array",
            "description": """Encode the categorical variable by One-hot encoding
            """,
            "items": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": """the source column with binary
                        encoding for the data."""
                    },
                    "prefix": {
                        "type": "string",
                        "description": "the new column name prefix."
                    },
                    "cats": {
                        "type": "array",
                        'items': {
                            "type": "integer"
                        },
                        "description": "an arrya of categories"
                    },
                    "prefix_sep": {
                        "type": "string",
                        "description": """the separator between the prefix
                        and the category.""",
                        "default": "_"
                    },
                    "dtype": {
                        "type": "string",
                        "description": "the dtype for the outputs",
                        "enum": ["float64", "float32", "int64", "int32"],
                        "default": "float64"
                    }
                }
            }
        }
        ui = {
        }
        input_meta = self.get_input_meta()
        if self.INPUT_PORT_NAME in input_meta:
            col_from_inport = input_meta[self.INPUT_PORT_NAME]
            enums = [col for col in col_from_inport.keys()]
            json['items']['properties']['column']['enum'] = enums
            return ConfSchema(json=json, ui=ui)
        else:
            return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        encode the categorical variables to one hot
        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        input_df = inputs[self.INPUT_PORT_NAME]
        for col in self.conf:
            input_df = input_df.one_hot_encoding(**col)
        return {self.OUTPUT_PORT_NAME: input_df}
