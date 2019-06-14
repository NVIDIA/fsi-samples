from gquant.dataframe_flow import Node
import cudf
import pandas as pd


class StockNameLoader(Node):

    def columns_setup(self):
        self.required = {}
        self.addition = {"asset": "int64",
                         "asset_name": "object"}
        self.deletion = None
        self.retention = None

    def process(self, inputs):
        """
        Load the csv file mapping stock id to symbol name into cudf DataFrame

        Arguments
        -------
         inputs: list
             empty list
        Returns
        -------
        cudf.DataFrame
        """

        name_df = pd.read_csv(self.conf['path'])[['SM_ID', 'SYMBOL']]
        output = cudf.from_pandas(name_df)
        # change the names
        output.columns = ["asset", 'asset_name']
        return output


if __name__ == "__main__":
    conf = {"path": "/home/yi/Projects/stocks/security_master.csv.gz"}
    loader = StockNameLoader("id0", conf, False, False)
    df3 = loader([])
