from gquant.dataframe_flow import Node


class SortNode(Node):

    def process(self, inputs):
        """
        Sort the input frames based on a list of columns, which are defined
        in the `keys` of the node's conf

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        input_df = inputs[0]
        return input_df.sort_values(self.conf['keys'])

    def columns_setup(self):
        self.delayed_process = True


if __name__ == "__main__":
    from gquant.dataloader.csvStockLoader import CsvStockLoader

    loader = CsvStockLoader("id0", {}, True, False)
    df = loader([])
    sf = SortNode("id2", {"keys": ["asset", 'datetime']})
    df2 = sf([df])
