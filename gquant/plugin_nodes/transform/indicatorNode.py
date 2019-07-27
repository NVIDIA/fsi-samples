from gquant.dataframe_flow import Node
import numpy as np
import gquant.cuindicator as ci


class IndicatorNode(Node):

    def columns_setup(self):
        self.required = {'indicator': 'int32'}
        self.addition = {}
        indicators = self.conf['indicators']
        for indicator in indicators:
            for col in indicator['columns']:
                self.required[col] = 'float64'
            if 'outputs' in indicator:
                for out in indicator['outputs']:
                    out_col = self._compose_name(indicator, [out])
                    self.addition[out_col] = 'float64'
            else:
                out_col = self._compose_name(indicator, [])
                self.addition[out_col] = 'float64'

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
        input_df = inputs[0]
        indicators = self.conf['indicators']
        out_cols = []
        for indicator in indicators:
            fun = getattr(ci, indicator['function'])
            parallel = [input_df['indicator']]
            data = [input_df[col] for col in indicator['columns']]
            ar = []
            if 'args' in indicator:
                ar = indicator['args']
            v = fun(*(parallel+data+ar))
            if isinstance(v, tuple) and 'outputs' in indicator:
                for out in indicator['outputs']:
                    out_col = self._compose_name(indicator, [out])
                    input_df[out_col] = getattr(v, out)
                    out_cols.append(out_col)
            else:
                out_col = self._compose_name(indicator, [])
                input_df[out_col] = v
                out_cols.append(out_col)
        # remove all the na elements, requires cudf>=0.8
        if "remove_na" in self.conf and self.conf["remove_na"]:
            na_element = input_df[out_cols[0]].isna()
            for i in range(1, len(out_cols)):
                na_element |= input_df[out_cols[i]].isna()
            input_df = input_df.iloc[np.where((~na_element).to_array())[0]]
        return input_df


if __name__ == "__main__":
    from gquant.plugin_nodes.dataloader.csvStockLoader import CsvStockLoader

    loader = CsvStockLoader("node_sort2", {}, True, False)
    df = loader.load_cache('.cache'+'/'+loader.uid+'.hdf5')
    conf = {
        "indicators": [
            {"function": "port_chaikin_oscillator",
             "columns": ["high", "low", "close", "volume"],
             "args": [10, 20]},
            {"function": "port_bollinger_bands",
             "columns": ["close"],
             "args": [10],
             "outputs": ["b1", "b2"]}
        ],
        "remove_na": True
    }
    inN = IndicatorNode("abc", conf)
    o = inN.process([df])
