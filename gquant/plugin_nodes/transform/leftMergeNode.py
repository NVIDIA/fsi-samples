from gquant.dataframe_flow import Node


class LeftMergeNode(Node):

    def process(self, inputs):
        """
        left merge the two dataframes in the inputs. the `on column` is defined
        in the `column` of the node's conf

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """

        df1 = inputs[0]
        df2 = inputs[1]
        return df1.merge(df2, on=self.conf['column'], how='left')

    def columns_setup(self):
        pass


if __name__ == "__main__":
    from gquant.dataloader.csvStockLoader import CsvStockLoader
    from gquant.dataloader.stockNameLoader import StockNameLoader

    loader = CsvStockLoader("id0", {}, True, False)
    df1 = loader([])
    sloader = StockNameLoader(
        "id1",
        {"path": "/home/yi/Projects/stocks/security_master.csv.gz"},
        False, False)
    df2 = sloader([])

    vf = LeftMergeNode("id2", {"column": "asset"})
    df3 = vf([df1, df2])
