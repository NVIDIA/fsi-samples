from gquant.dataframe_flow import Node
from .returnFeatureNode import ReturnFeatureNode


class DropNode(Node):

    def columns_setup(self):
        self.delayed_process = True
        self.deletion = {"@columns": None}

    def process(self, inputs):
        """
        Drop a few columns from the dataframe that are defined in the `columns`
        in the nodes' conf

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        input_df = inputs[0]
        column_names = self.conf['columns']
        return input_df.drop(column_names, axis=1)


if __name__ == "__main__":
    from gquant.dataloader.csvStockLoader import CsvStockLoader

    loader = CsvStockLoader("id0", {}, True, False)
    df = loader([])
    df = df.sort_values(["asset", 'datetime'])
    sf = ReturnFeatureNode("id2", {})
    df2 = sf([df])
