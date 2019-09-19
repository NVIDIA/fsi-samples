from gquant.dataframe_flow import Node


class SimpleAveragePortOpt(Node):

    def columns_setup(self):
        self.required = {"datetime": "date",
                         "strategy_returns": "float64",
                         "asset": "int64"}
        self.retention = {"datetime": "date",
                          "strategy_returns": "float64"}

    def process(self, inputs):
        """
        Average the strategy returns for all the assets.

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        input_df = inputs[0]
        port = input_df[['datetime', 'strategy_returns']] \
            .groupby(['datetime']).mean().reset_index()
        port.columns = ['datetime', 'strategy_returns']
        return port

if __name__ == "__main__":
    pass
