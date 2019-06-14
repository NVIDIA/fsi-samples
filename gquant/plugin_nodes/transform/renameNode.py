from gquant.dataframe_flow import Node
from .averageNode import AverageNode


class RenameNode(Node):

    def process(self, inputs):
        """
        Rename the column name in the datafame from `old` to `new` defined in
        the node's conf

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        input_df = inputs[0]
        new_column = self.conf['new']
        old_column = self.conf['old']
        return input_df.rename(columns={old_column: new_column})

    def columns_setup(self):
        self.rename = {"@old": "@new"}


if __name__ == "__main__":
    from gquant.dataloader.csvStockLoader import CsvStockLoader

    loader = CsvStockLoader("id0", {}, True, False)
    df = loader([])
    vf = AverageNode("id1", {"column": "volume"})
    df2 = vf([df])
