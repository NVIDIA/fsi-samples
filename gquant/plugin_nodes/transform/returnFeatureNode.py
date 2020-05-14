from gquant.dataframe_flow import Node
import gquant.cuindicator as ci
from numba import cuda
import numpy as np


def mask_returns(close, indicator):
    print(len(close), cuda.threadIdx.x, cuda.blockDim.x, len(indicator))
    for i in range(cuda.threadIdx.x, len(close), cuda.blockDim.x):
        if i == 0:
            indicator[i] = np.nan
        else:
            indicator[i] = 0


def clean(df):
    df.iloc[0] = np.nan
    return df


class ReturnFeatureNode(Node):

    def columns_setup(self):
        self.delayed_process = True

        self.required = {"close": "float64",
                         "asset": "int64"}
        self.addition = {"returns": "float64"}

    def process(self, inputs):
        """
        Add the rate of of return column based on the `close` price for each
        of the asset in the dataframe. The result column is named as `returns`

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        input_df = inputs[0]
        input_df = input_df.reset_index().drop('index')
        val = ci.rate_of_change(input_df['close'], 2).fillna(0.0)
        input_df['returns'] = val
        input_df = input_df.groupby(["asset"], method='cudf') \
            .apply_grouped(mask_returns,
                           incols=['close'],
                           outcols={'indicator': 'int64'},
                           tpb=1)
        return input_df.nans_to_nulls().dropna().drop('indicator')


class CpuReturnFeatureNode(ReturnFeatureNode):

    def process(self, inputs):
        """
        Add the rate of of return column based on the `close` price for each
        of the asset in the dataframe. The result column is named as `returns`

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        input_df = inputs[0]
        shifted = input_df['close'].shift(1)
        input_df['returns'] = (input_df['close'] - shifted) / shifted
        input_df['returns'] = input_df['returns'].fillna(0.0)
        input_df = input_df.groupby('asset').apply(clean)
        return input_df.dropna()


if __name__ == "__main__":
    from gquant.dataloader.csvStockLoader import CsvStockLoader

    loader = CsvStockLoader("id0", {}, True, False)
    df = loader([])
    df = df.sort_values(["asset", 'datetime'])
    sf = ReturnFeatureNode("id2", {})
    df2 = sf([df])
