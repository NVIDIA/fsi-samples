from gquant.dataframe_flow import Node
import math
import dask_cudf


class SharpeRatioNode(Node):

    def columns_setup(self):
        self.required = {"strategy_returns": "float64"}
        self.retentation = {}

    def process(self, inputs):
        """
        Compute the yearly Sharpe Ratio from the input dataframe
        `strategy_returns` column. Assume it is daily return. Asumes
        252 trading days per year.

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        float
            the sharpe ratio
        """

        input_df = inputs[0]
        if isinstance(input_df,  dask_cudf.DataFrame):
            input_df = input_df.compute()  # get the computed value
        daily_mean = input_df['strategy_returns'].mean()
        daily_std = input_df['strategy_returns'].std()
        return daily_mean / daily_std * math.sqrt(252)


if __name__ == "__main__":
    from gquant.dataloader.csvStockLoader import CsvStockLoader
    from gquant.transform.averageNode import AverageNode
    from gquant.analysis.outCsvNode import OutCsvNode

    loader = CsvStockLoader("id0", {}, True, False)
    df = loader([])
    vf = AverageNode("id1", {"column": "volume"})
    df2 = vf([df])
    o = OutCsvNode("id3", {"path": "o.csv"})
    o([df2])
