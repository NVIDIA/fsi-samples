from gquant.dataframe_flow import Node
from .volumeFilterNode import VolumeFilterNode


class ValueFilterNode(Node):

    def columns_setup(self):
        self.required = {"asset": "int64"}

    def process(self, inputs):
        """
        filter the dataframe based on a list of min/max values. The node's
        conf is a list of column criteria. It defines the column name in
        'column`, the min value in `min` and the max value in `max`.

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """

        input_df = inputs[0]
        str_list = []
        for column_item in self.conf:
            column_name = column_item['column']
            if 'min' in column_item:
                minValue = column_item['min']
                str_item = '%s >= %f' % (column_name, minValue)
                str_list.append(str_item)
            if 'max' in column_item:
                maxValue = column_item['max']
                str_item = '%s <= %f' % (column_name, maxValue)
                str_list.append(str_item)
        input_df = input_df.query(" and ".join(str_list))
        return input_df


if __name__ == "__main__":
    from gquant.dataloader.csvStockLoader import CsvStockLoader

    loader = CsvStockLoader("id0", {}, True, False)
    df = loader([])
    vf = VolumeFilterNode("id1", {"min": 50.0})
    df2 = vf([df])
