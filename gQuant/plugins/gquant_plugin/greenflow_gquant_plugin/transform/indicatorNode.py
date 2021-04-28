import copy
from greenflow.dataframe_flow import (ConfSchema, PortsSpecSchema)
from greenflow.dataframe_flow.metaSpec import MetaDataSchema
from greenflow.dataframe_flow import Node
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin
from ..node_hdf_cache import NodeHDFCacheMixin

from .. import cuindicator as ci

__all__ = ['IndicatorNode']


IN_DATA = {
    "port_exponential_moving_average": {
        "function": "port_exponential_moving_average",
        "columns": ["close"],
        "args": [2]
    },
    "port_moving_average": {
        "function": "port_moving_average",
        "columns": ["close"],
        "args": [2]
    },
    "port_rate_of_change": {
        "function": "port_rate_of_change",
        "columns": ["close"],
        "args": [2]
    },
    "port_diff": {"function": "port_diff",
                  "columns": ["close"],
                  "args": [-1]
                  },
    "port_trix": {
        "function": "port_trix",
        "columns": ["close"],
        "args": [2]
    },
    "port_average_directional_movement_index": {
        "function": "port_average_directional_movement_index",
        "columns": ["high", "low", "close"],
        "args": [2, 3]
    },
    "port_donchian_channel": {
        "function": "port_donchian_channel",
        "columns": ["high", "low"],
        "args": [2]
    },
    "port_fractional_diff": {"function": "port_fractional_diff",
                             "columns": ["close"],
                             "args": [0.9]
                             },
    "port_chaikin_oscillator": {"function": "port_chaikin_oscillator",
                                "columns": ["high", "low", "close", "volume"],
                                "args": [2, 3]
                                },
    "port_bollinger_bands": {"function": "port_bollinger_bands",
                             "columns": ["close"],
                             "args": [2],
                             "outputs": ["b1", "b2"]
                             },
    "port_macd": {"function": "port_macd",
                  "columns": ["close"],
                  "args": [2, 3],
                  "outputs": ["MACDsign", "MACDdiff"]
                  },
    "port_relative_strength_index": {
        "function": "port_relative_strength_index",
        "columns": ["high", "low"],
        "args": [2],
    },
    "port_average_true_range": {"function": "port_average_true_range",
                                "columns": ["high", "low", "close"],
                                "args": [2],
                                },
    "port_stochastic_oscillator_k": {
        "function": "port_stochastic_oscillator_k",
        "columns": ["high", "low", "close"],
        "args": [],
    },
    "port_stochastic_oscillator_d": {
        "function": "port_stochastic_oscillator_d",
        "columns": ["high", "low", "close"],
        "args": [2],
    },
    "port_money_flow_index": {
        "function": "port_money_flow_index",
        "columns": ["high", "low", "close", "volume"],
        "args": [2],
    },
    "port_force_index": {"function": "port_force_index",
                         "columns": ["close", "volume"],
                         "args": [2],
                         },
    "port_ultimate_oscillator": {"function": "port_ultimate_oscillator",
                                 "columns": ["high", "low", "close"],
                                 "args": [],
                                 },
    "port_accumulation_distribution": {
        "function": "port_accumulation_distribution",
        "columns": ["high", "low", "close", "volume"],
        "args": [2],
    },
    "port_commodity_channel_index": {
        "function": "port_commodity_channel_index",
        "columns": ["high", "low", "close"],
        "args": [2],
    },
    "port_on_balance_volume": {"function": "port_on_balance_volume",
                               "columns": ["close", "volume"],
                               "args": [2],
                               },
    "port_vortex_indicator": {"function": "port_vortex_indicator",
                              "columns": ["high", "low", "close"],
                              "args": [2],
                              },
    "port_kst_oscillator": {"function": "port_kst_oscillator",
                            "columns": ["close"],
                            "args": [3, 4, 5, 6, 7, 8, 9, 10],
                            },
    "port_mass_index": {"function": "port_mass_index",
                        "columns": ["high", "low"],
                        "args": [2, 3],
                        },
    "port_true_strength_index": {"function": "port_true_strength_index",
                                 "columns": ["close"],
                                 "args": [2, 3],
                                 },
    "port_ease_of_movement": {"function": "port_ease_of_movement",
                              "columns": ["high", "low", "volume"],
                              "args": [2],
                              },
    "port_coppock_curve": {"function": "port_coppock_curve",
                           "columns": ["close"],
                           "args": [2],
                           },
    "port_keltner_channel": {"function": "port_keltner_channel",
                             "columns": ["high", "low", "close"],
                             "args": [2],
                             "outputs": ["KelChD", "KelChM", "KelChU"]
                             },
    "port_ppsr": {"function": "port_ppsr",
                  "columns": ["high", "low", "close"],
                  "args": [],
                  "outputs": ["PP", "R1", "S1", "R2", "S2", "R3", "S3"]
                  },
    "port_shift": {"function": "port_shift",
                   "columns": ["returns"],
                   "args": [-1]
                   }
}


class IndicatorNode(TemplateNodeMixin, NodeHDFCacheMixin, Node):

    def init(self):
        TemplateNodeMixin.init(self)
        self.delayed_process = True
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
        cols_required = {'indicator': 'int32'}
        addition = {}
        if 'indicators' in self.conf:
            indicators = self.conf['indicators']
            for indicator in indicators:
                functionId = indicator['function']
                conf = copy.deepcopy(IN_DATA[functionId])
                if 'args' in indicator:
                    if len(conf['args']) != 0:
                        conf['args'] = indicator['args']
                if 'columns' in indicator:
                    conf['columns'] = indicator['columns']
                for col in conf['columns']:
                    cols_required[col] = 'float64'
                if 'outputs' in conf:
                    for out in conf['outputs']:
                        out_col = self._compose_name(conf, [out])
                        addition[out_col] = 'float64'
                else:
                    out_col = self._compose_name(conf, [])
                    addition[out_col] = 'float64'
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

    def _compose_name(self, indicator, outname=[]):
        name = indicator['function']
        args_name = []
        if 'args' in indicator:
            args_name = [str(i) for i in indicator['args']]

        splits = [i.upper() for i in name.split('_') if i != 'port']
        if len(splits) > 2:
            splits = [i[0] for i in splits] + outname + args_name
        elif len(splits) == 2:
            splits = [i[0:2] for i in splits] + outname + args_name
        else:
            splits = [splits[0]] + outname + args_name
        return "_".join(splits)

    def conf_schema(self):
        json = {
            "title": "Technical Indicator Node configure",
            "type": "object",
            "description": """Add technical indicators to the dataframe.
            "remove_na" decides whether we want to remove the NAs
            from the technical indicators""",
            "properties": {
                "indicators":  {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "anyOf": [
                        ]
                    },
                    "description": """A list of indicators to be included"""
                },
                "remove_na":  {
                    "type": "boolean",
                    "description": """Remove the NAs from the technical
                     indicators?""",
                    "enum": [True, False],
                    "default": True
                }

            },
            "required": ["remove_na"],
        }
        input_meta = self.get_input_meta()
        enums = []
        if self.INPUT_PORT_NAME in input_meta:
            col_from_inport = input_meta[self.INPUT_PORT_NAME]
            enums = [col for col in col_from_inport.keys()]
        for key in IN_DATA.keys():
            fun_name = " ".join(key.split('_')[1:])
            out = {
                "function": {
                    "title": fun_name,
                    "enum": [key],
                    "default": key,
                },
            }
            args = {
                "type": "array",
                "items": []
            }
            columns = {
                "type": "array",
                "items": []
            }
            for arg in range(len(IN_DATA[key]['args'])):
                item = {
                    "type": "number",
                    "title": "parameter {}".format(arg+1),
                    "default": IN_DATA[key]['args'][arg]
                }
                args['items'].append(item)
            for arg in range(len(IN_DATA[key]['columns'])):
                item = {
                    "type": "string",
                    "default": IN_DATA[key]['columns'][arg]
                }
                if len(enums) > 0:
                    item['enum'] = enums
                columns['items'].append(item)
            if (len(IN_DATA[key]['args']) > 0):
                out['args'] = args
            if (len(IN_DATA[key]['columns']) > 0):
                out['columns'] = columns
            obj = {"type": "object", "properties": out, "title": fun_name}
            json['properties']['indicators']['items']['anyOf'].append(obj)
        ui = {}
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        Add technical indicators to the dataframe.
        All technical indicators are defined in the self.conf
        "remove_na" in self.conf decides whether we want to remove the NAs
        from the technical indicators

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        input_df = inputs[self.INPUT_PORT_NAME]
        indicators = self.conf['indicators']
        for indicator in indicators:
            functionId = indicator['function']
            conf = copy.deepcopy(IN_DATA[functionId])
            if 'args' in indicator:
                #  a bug work around to ignore the numbers from the client
                if len(conf['args']) != 0:
                    conf['args'] = indicator['args']
            if 'columns' in indicator:
                conf['columns'] = indicator['columns']
            fun = getattr(ci, indicator['function'])
            parallel = [input_df['indicator']]
            data = [input_df[col] for col in conf['columns']]
            ar = []
            if 'args' in conf:
                ar = conf['args']
            v = fun(*(parallel+data+ar))
            if isinstance(v, tuple) and 'outputs' in conf:
                for out in conf['outputs']:
                    out_col = self._compose_name(conf, [out])
                    val = getattr(v, out)
                    val.index = input_df.index
                    input_df[out_col] = val
            else:
                if isinstance(v, tuple):
                    v = v[0]
                out_col = self._compose_name(conf, [])
                v.index = input_df.index
                input_df[out_col] = v
        # remove all the na elements, requires cudf>=0.8
        if "remove_na" in self.conf and self.conf["remove_na"]:
            input_df = input_df.nans_to_nulls().dropna()
        return {self.OUTPUT_PORT_NAME: input_df}
