from gquant.dataframe_flow import Node
import datetime


class DatetimeFilterNode(Node):

    def process(self, inputs):
        """
        select the data based on an range of datetime, which is defined in
        `beg` and `end` in the nodes' conf

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        beg_date = datetime.datetime.strptime(self.conf['beg'], '%Y-%m-%d')  # noqa: F841
        end_date = datetime.datetime.strptime(self.conf['end'], '%Y-%m-%d')  # noqa: F841
        return df.query('datetime<@end_date and datetime>=@beg_date')

    def columns_setup(self):
        self.required = {"datetime": "datetime64[ms]"}


if __name__ == "__main__":
    from gquant.dataloader.csvStockLoader import CsvStockLoader

    loader = CsvStockLoader("node_csvdata", {}, True, False)
    df = loader([])
    sf = DatetimeFilterNode("id2", {"beg": '2011-01-01', "end": '2012-01-01'})
    df2 = sf([df])
