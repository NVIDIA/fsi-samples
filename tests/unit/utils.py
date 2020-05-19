import numpy as np


def make_orderer():
    """Keep tests in order"""
    order = {}

    def ordered(f):
        order[f.__name__] = len(order)
        return f

    def compare(a, b):
        return [1, -1][order[a] < order[b]]

    return ordered, compare


def error_function(gpu_series, result_series):
    """
    utility function to compare GPU array vs CPU array
    Parameters
    ------
    gpu_series: cudf.Series
        GPU computation result series
    result_series: pandas.Series
        Pandas computation result series

    Returns
    -----
    double
        maximum error of the two arrays
    """
    gpu_arr = gpu_series.to_array(fillna='pandas')
    pan_arr = result_series.values
    gpu_arr = gpu_arr[~np.isnan(gpu_arr) & ~np.isinf(gpu_arr)]
    pan_arr = pan_arr[~np.isnan(pan_arr) & ~np.isinf(pan_arr)]
    err = np.abs(gpu_arr - pan_arr).max()
    return err


def error_function_index(gpu_series, result_series):
    """
    utility function to compare GPU array vs CPU array
    Parameters
    ------
    gpu_series: cudf.Series
        GPU computation result series
    result_series: pandas.Series
        Pandas computation result series

    Returns
    -----
    double
        maximum error of the two arrays
    int
        maximum index value diff
    """
    err = error_function(gpu_series, result_series)
    error_index = np.abs(gpu_series.index.to_array() -
                         result_series.index.values).max()
    return err, error_index
