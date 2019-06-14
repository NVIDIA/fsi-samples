from gquant.dataframe_flow import Node


class AverageNode(Node):

    def process(self, inputs):
        """
        Compute the average value of the key column which is defined in the
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
        average_column = self.conf['column']
        volume_df = input_df[[average_column, "asset"]] \
            .groupby(["asset"]).mean().reset_index()
        volume_df.columns = ['asset', average_column]
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
