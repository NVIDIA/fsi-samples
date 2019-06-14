from gquant.dataframe_flow import Node


class AssetFilterNode(Node):

    def process(self, inputs):
        """
        select the asset based on asset id, which is defined in `asset` in the
        nodes' conf

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        input_df = inputs[0]
        output_df = input_df.query('asset==%s' % self.conf["asset"])
        return output_df

    def columns_setup(self):
        self.required = {"asset": "int64"}


if __name__ == "__main__":
    from gquant.dataloader.csvStockLoader import CsvStockLoader
    loader = CsvStockLoader("node_csvdata", {}, True, False)
    df = loader([])
    sf = AssetFilterNode("id2", {"asset": 22123})
    df2 = sf([df])
