from gquant.dataframe_flow import Node


class VolumeFilterNode(Node):

    def columns_setup(self):
        self.required = {"asset": "int64",
                         "volume": "float64"}
        self.addition = {"mean_volume": "float64"}

    def process(self, inputs):
        """
        filter the dataframe based on the min and max values of the average
        volume for each fo the assets.

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """

        input_df = inputs[0]
        volume_df = input_df[['volume', "asset"]].groupby(
                ["asset"]).mean().reset_index()
        volume_df.columns = ["asset", 'mean_volume']
        merged = input_df.merge(volume_df, on="asset", how='left')
        if 'min' in self.conf:
            minVolume = self.conf['min']
            merged = merged.query('mean_volume >= %f' % (minVolume))
        if 'max' in self.conf:
            maxVolume = self.conf['max']
            merged = merged.query('mean_volume <= %f' % (maxVolume))
        return merged


if __name__ == "__main__":
    from gquant.dataloader.csvStockLoader import CsvStockLoader

    loader = CsvStockLoader("id0", {}, True, False)
    df = loader([])
    vf = VolumeFilterNode("id1", {"min": 50.0})
    df2 = vf([df])
