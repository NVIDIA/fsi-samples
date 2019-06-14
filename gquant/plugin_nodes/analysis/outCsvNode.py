from gquant.dataframe_flow import Node
import dask_cudf


class OutCsvNode(Node):

    def columns_setup(self):
        self.required = {}

    def process(self, inputs):
        """
        dump the input datafram to the resulting csv file.
        the output filepath is defined as `path` in the `conf`.
        if only a subset of columns is needed for the csv file, enumerate the
        columns in the `columns` of the `conf`

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """

        input_df = inputs[0]
        if isinstance(input_df,  dask_cudf.DataFrame):
            input_df = input_df.compute()  # get the computed value
        if 'columns' in self.conf:
            input_df = input_df[self.conf['columns']]
        input_df.to_pandas().to_csv(self.conf['path'], index=False)
        return input_df


if __name__ == "__main__":
    from gquant.dataloader.csvStockLoader import CsvStockLoader
    from gquant.transform.averageNode import AverageNode

    loader = CsvStockLoader("id0", {}, True, False)
    df = loader([])
    vf = AverageNode("id1", {"column": "volume"})
    df2 = vf([df])
    o = OutCsvNode("id3", {"path": "o.csv"})
    o([df2])
