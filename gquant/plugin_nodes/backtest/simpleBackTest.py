from gquant.dataframe_flow import Node


class SimpleBackTestNode(Node):

    def columns_setup(self):
        self.required = {"signal": "float64",
                         "returns": "float64"}
        self.addition = {"strategy_returns": "float64"}

    def process(self, inputs):
        """
        compute the `strategy_returns` by assuming invest `signal` amount of
        dollars for each of the time step.

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        input_df = inputs[0]
        input_df['strategy_returns'] = input_df['signal'] * input_df['returns']
        return input_df

if __name__ == "__main__":
    from gquant.dataloader.csvStockLoader import CsvStockLoader
    from gquant.transform.assetFilterNode import AssetFilterNode
    from gquant.transform.sortNode import SortNode
    from gquant.transform.returnFeatureNode import ReturnFeatureNode
    from gquant.strategy import MovingAverageStrategyNode

    loader = CsvStockLoader("node_csvdata", {}, True, False)
    df = loader([])
    sf = AssetFilterNode("id2", {"asset": 22123})
    df2 = sf([df])
    sf2 = SortNode("id3", {"keys": ["asset", 'datetime']})
    df3 = sf2([df2])
    sf3 = ReturnFeatureNode('id4', {})
    df4 = sf3([df3])
    sf4 = MovingAverageStrategyNode('id5', {'fast': 5, 'slow': 10})
    df5 = sf4([df4])
    sf5 = SimpleBackTestNode('id6', {})
    df6 = sf5([df5])
