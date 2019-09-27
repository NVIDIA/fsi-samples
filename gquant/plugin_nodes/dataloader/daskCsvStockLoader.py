from gquant.dataframe_flow import Node
from .csvStockLoader import CsvStockLoader
import dask_cudf


class DaskCsvStockLoader(Node):

    def columns_setup(self):
        self.required = {}
        self.addition = {"datetime": "datetime64[ms]",
                         "asset": "int64",
                         "volume": "float64",
                         "close": "float64",
                         "open": "float64",
                         "high": "float64",
                         "low": "float64"}
        self.deletion = None
        self.retention = None

    def process(self, inputs):
        """
        Load the end of day multiple stock CSV files into distributed dask cudf
        dataframe

        Arguments
        -------
         inputs: list
             empty list
        Returns
        -------
        dask_cudf.DataFrame
        """

        gdf = dask_cudf.read_csv(self.conf['path']+'/*.csv')
        return gdf


if __name__ == "__main__":
    conf = {"path": "/home/yi/Projects/stocks/stock_price_hist.csv.gz"}
    loader = CsvStockLoader("node_csvdata", conf, False, True)
    df = loader([])
