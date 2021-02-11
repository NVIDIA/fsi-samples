import datetime
from greenflow.dataframe_flow import Node
from .._port_type_node import _PortTypesMixin
from greenflow.dataframe_flow.portsSpecSchema import ConfSchema


__all__ = ['DatetimeFilterNode']


class DatetimeFilterNode(_PortTypesMixin, Node):
    """
    A node that is used to select datapoints based on range of time.
    conf["beg"] defines the beginning of the date inclusively and
    conf["end"] defines the end of the date exclusively.
    all the date strs are in format of "Y-m-d".

    """

    def init(self):
        _PortTypesMixin.init(self)
        self.INPUT_PORT_NAME = 'stock_in'
        self.OUTPUT_PORT_NAME = 'stock_out'

    def meta_setup(self):
        cols_required = {"datetime": "date"}
        return _PortTypesMixin.meta_setup(self,
                                          required=cols_required)

    def ports_setup(self):
        return _PortTypesMixin.ports_setup(self)

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
        beg_date = \
            datetime.datetime.strptime(self.conf['beg'], '%Y-%m-%d')
        end_date = \
            datetime.datetime.strptime(self.conf['end'], '%Y-%m-%d')
        df = df.query('datetime<@end_date and datetime>=@beg_date',
                      local_dict={'beg_date': beg_date,
                                  'end_date': end_date})
        return {self.OUTPUT_PORT_NAME: df}
