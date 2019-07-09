from gquant.dataframe_flow import Node
from .returnFeatureNode import ReturnFeatureNode
import gquant.cuindicator as ci


class IndicatorNode(Node):

    def columns_setup(self):
        self.delayed_process = True
        self.deletion = {"@columns": None}

    def process(self, inputs):
        """
        Drop a few columns from the dataframe that are defined in the `columns`
        in the nodes' conf

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        input_df = inputs[0]
        indicators = self.conf['indicators']
        for indicator in indicators:
            fun = getattr(ci, indicator['function'])
            data = [input_df[col] for col in indicator['columns']]
            ar = indicator['args']
            v = fun(*(data+ar))
        input_df['t'] = v
        return input_df


if __name__ == "__main__":
    from gquant.plugin_nodes.dataloader.csvStockLoader import CsvStockLoader

    loader = CsvStockLoader("node_csvdata", {}, True, False)
    df = loader.load_cache('.cache'+'/'+loader.uid+'.hdf5')
    conf = {
        "indicators":[
            {"function": "chaikin_oscillator",
             "columns": ["high", "low", "close", "volume"],
             "args": [10, 20],
            }
        ]
    }
    inN = IndicatorNode("abc", conf)
    o = inN.process([df])
    # df = df.sort_values(["asset", 'datetime'])
    # sf = ReturnFeatureNode("id2", {})
    # df2 = sf([df])
