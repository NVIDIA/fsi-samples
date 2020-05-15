from gquant.dataframe_flow import Node
from .returnFeatureNode import ReturnFeatureNode
from numba import cuda


def indicator_fun(indicator):
    for i in range(cuda.threadIdx.x, indicator.size, cuda.blockDim.x):
        if i == 0:
            indicator[i] = 1
        else:
            indicator[i] = 0


def cpu_indicator_fun(df):
    df['indicator'] = 0
    df['indicator'].values[0] = 1
    return df


class AssetIndicatorNode(Node):

    def columns_setup(self):
        self.delayed_process = True
        self.required = {"asset": "int64"}
        self.addition = {"indicator": "int32"}

    def process(self, inputs):
        """
        Add the indicator column in the dataframe which set 1 at the beginning
        of the each of the assets

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """

        input_df = inputs[0]
        input_df = input_df.reset_index(drop=True)
        input_df = input_df.groupby(["asset"], method='cudf') \
            .apply_grouped(indicator_fun,
                           incols=[],
                           outcols={'indicator': 'int32'},
                           tpb=256)
        return input_df


class CpuAssetIndicatorNode(AssetIndicatorNode):

    def process(self, inputs):
        """
        Add the indicator column in the dataframe which set 1 at the beginning
        of the each of the assets

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        input_df = inputs[0]
        input_df = input_df.groupby("asset").apply(cpu_indicator_fun)
        return input_df


if __name__ == "__main__":
    from gquant.dataloader.csvStockLoader import CsvStockLoader

    loader = CsvStockLoader("id0", {}, True, False)
    df = loader([])
    df = df.sort_values(["asset", 'datetime'])
    sf = ReturnFeatureNode("id2", {})
    df2 = sf([df])
