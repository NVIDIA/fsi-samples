from greenflow.dataframe_flow import Node
from greenflow.dataframe_flow.portsSpecSchema import (ConfSchema,
                                                      PortsSpecSchema)
from greenflow.dataframe_flow.metaSpec import MetaDataSchema
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin
from ..node_hdf_cache import NodeHDFCacheMixin

__all__ = ['ValueFilterNode']


class ValueFilterNode(TemplateNodeMixin, NodeHDFCacheMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.INPUT_PORT_NAME = 'in'
        self.OUTPUT_PORT_NAME = 'out'
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
        cols_required = {"asset": "int64"}
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
        json = {
            "title": "Value Filter Node configure",
            "type": "array",
            "description": """Filter the dataframe based on a list of
            min/max values.""",
            "items": {
                "type": "object",
                "properties": {
                    "column":  {
                        "type": "string",
                        "description": "dataframe column to be filered on"
                    },
                    "min": {
                        "type": "number",
                        "description": "min value, inclusive"
                    },
                    "max": {
                        "type": "number",
                        "description": "max value, inclusive"
                    }
                }
            }
        }
        ui = {}
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
        filter the dataframe based on a list of min/max values. The node's
        conf is a list of column criteria. It defines the column name in
        'column`, the min value in `min` and the max value in `max`.

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """

        input_df = inputs[self.INPUT_PORT_NAME]
        str_list = []
        for column_item in self.conf:
            column_name = column_item['column']
            if 'min' in column_item:
                minValue = column_item['min']
                str_item = '%s >= %f' % (column_name, minValue)
                str_list.append(str_item)
            if 'max' in column_item:
                maxValue = column_item['max']
                str_item = '%s <= %f' % (column_name, maxValue)
                str_list.append(str_item)
        input_df = input_df.query(" and ".join(str_list))
        return {self.OUTPUT_PORT_NAME: input_df}
