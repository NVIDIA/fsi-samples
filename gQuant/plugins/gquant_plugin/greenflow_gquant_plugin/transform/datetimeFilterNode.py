import datetime
from greenflow.dataframe_flow import Node, PortsSpecSchema
from greenflow.dataframe_flow.portsSpecSchema import ConfSchema
from greenflow.dataframe_flow.metaSpec import MetaDataSchema
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin
from ..node_hdf_cache import NodeHDFCacheMixin

__all__ = ['DatetimeFilterNode']


class DatetimeFilterNode(TemplateNodeMixin, NodeHDFCacheMixin, Node):
    """
    A node that is used to select datapoints based on range of time.
    conf["beg"] defines the beginning of the date inclusively and
    conf["end"] defines the end of the date exclusively.
    all the date strs are in format of "Y-m-d".

    """

    def init(self):
        TemplateNodeMixin.init(self)
        self.INPUT_PORT_NAME = 'stock_in'
        self.OUTPUT_PORT_NAME = 'stock_out'
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
                port_type: "${port:stock_in}"
            }
        }
        addition = {}
        cols_required = {"datetime": "datetime64[ns]"}
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
            "title": "Asset indicator configure",
            "type": "object",
            "description": """Select the data based on an range of datetime""",
            "properties": {
                "beg":  {
                    "type": "string",
                    "description": """start date, inclusive"""
                },
                "end":  {
                    "type": "string",
                    "description": """end date, exclusive"""
                }
            },
            "required": ["beg", "end"],
        }
        ui = {
            "beg": {"ui:widget": "alt-date",
                    "ui:options": {
                        "yearsRange": [1985, 2025],
                        "hideNowButton": True,
                        "hideClearButton": True,
                    }
                    },
            "end": {"ui:widget": "alt-date",
                    "ui:options": {
                        "yearsRange": [1985, 2025],
                        "hideNowButton": True,
                        "hideClearButton": True,
                    }
                    }
        }
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        Select the data based on an range of datetime, which is defined in
        `beg` and `end` in the nodes' conf

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        df = inputs[self.INPUT_PORT_NAME]
        beg_date = datetime.datetime.strptime(self.conf['beg'],
                                              '%Y-%m-%dT%H:%M:%S.%fZ')
        end_date = datetime.datetime.strptime(self.conf['end'],
                                              '%Y-%m-%dT%H:%M:%S.%fZ')
        df = df.query('datetime<@end_date and datetime>=@beg_date',
                      local_dict={
                          'beg_date': beg_date,
                          'end_date': end_date
                      })
        return {self.OUTPUT_PORT_NAME: df}
