from gquant.dataframe_flow import Node
import datetime


__all__ = ['DatetimeFilterNode']


class DatetimeFilterNode(Node):
    """
    A node that is used to select datapoints based on range of time.
    conf["beg"] defines the beginning of the date inclusively and
    conf["end"] defines the end of the date exclusively.
    all the date strs are in format of "Y-m-d".

    """

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
        df = inputs[0]
        beg_date = \
            datetime.datetime.strptime(self.conf['beg'], '%Y-%m-%d')
        end_date = \
            datetime.datetime.strptime(self.conf['end'], '%Y-%m-%d')
        return df.query('datetime<@end_date and datetime>=@beg_date',
                        local_dict={'beg_date': beg_date,
                                    'end_date': end_date})

    def columns_setup(self):
        self.required = {"datetime": "datetime64[ms]"}


if __name__ == "__main__":
    from gquant.dataloader.csvStockLoader import CsvStockLoader

    loader = CsvStockLoader("node_csvdata", {}, True, False)
    df = loader([])
    sf = DatetimeFilterNode("id2", {"beg": '2011-01-01', "end": '2012-01-01'})
    df2 = sf([df])
