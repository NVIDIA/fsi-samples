from gquant.dataframe_flow import Node
from .averageNode import AverageNode


class MaxNode(Node):

    def process(self, inputs):
        """
        Compute the maximum value of the key column which is defined in the
        `column` of the node's conf

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        input_df = inputs[0]
        max_column = self.conf['column']
        volume_df = input_df[[max_column,
                              "asset"]].groupby(["asset"]).max().reset_index()
        volume_df.columns = ['asset', max_column]
        return volume_df

    def columns_setup(self):
        self.required = {"asset": "int64"}
        self.retention = {"@column": "float64",
                          "asset": "int64"}


if __name__ == "__main__":
    from gquant.dataloader.csvStockLoader import CsvStockLoader

    loader = CsvStockLoader("id0", {}, True, False)
    df = loader([])
    vf = AverageNode("id1", {"column": "volume"})
    df2 = vf([df])
