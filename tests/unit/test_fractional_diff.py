'''
Fractional differencing Unit Tests

To run unittests:

# Using standard library unittest

python -m unittest -v
python -m unittest tests/unit/test_fractional_diff.py -v

or

python -m unittest discover <test_directory>
python -m unittest discover -s <directory> -p 'test_*.py'

# Using pytest
# "conda install pytest" or "pip install pytest"
pytest -v tests
pytest -v tests/unit/test_fractional_diff.py

'''
import pandas as pd
import unittest
import cudf
import os
from gquant.dataframe_flow.task import load_modules
load_modules(os.getenv('MODULEPATH')+'/rapids_modules/')
from rapids_modules.cuindicator import (fractional_diff, get_weights_floored,
                                        port_fractional_diff)
import numpy as np
from .utils import make_orderer
import warnings

ordered, compare = make_orderer()
unittest.defaultTestLoader.sortTestMethodsUsing = compare


def frac_diff(df, d, floor=1e-3):
    r"""Fractionally difference time series via CPU.
    code is copied from
    https://github.com/ritchieng/fractional_differencing_gpu/blob/master/notebooks/gpu_fractional_differencing.ipynb

    Args:
        df (pd.DataFrame): dataframe of raw time series values.
        d (float): differencing value from 0 to 1 where > 1 has no FD.
        floor (float): minimum value of weights, ignoring anything smaller.
    """
    # Get weights window
    weights = get_weights_floored(d=d, num_k=len(df), floor=floor)
    weights_window_size = len(weights)

    # Reverse weights
    weights = weights[::-1]

    # Blank fractionally differenced series to be filled
    df_fd = []

    # Slide window of time series,
    # to calculated fractionally differenced values
    # per window
    for idx in range(weights_window_size, df.shape[0]):
        # Dot product of weights and original values
        # to get fractionally differenced values
        # date_idx = df.index[idx]
        df_fd.append(np.dot(weights.T,
                            df.iloc[idx - weights_window_size:idx]).item())

    # Return FD values and weights
    df_fd = pd.DataFrame(df_fd)

    return df_fd, weights


class TestFracDiff(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings('ignore', message='numpy.ufunc size changed')
        array_len = int(1e4)
        random_array = np.random.rand(array_len)
        df = cudf.DataFrame()
        df['in'] = random_array

        pdf = pd.DataFrame()
        pdf['in'] = random_array

        # ignore importlib warnings.
        self._pandas_data = pdf
        self._cudf_data = df

        # data set for multiple assets
        size = 200
        half = size // 2
        self.size = size
        self.half = half
        np.random.seed(10)
        random_array = np.random.rand(size)
        indicator = np.zeros(size, dtype=np.int32)
        indicator[0] = 1
        indicator[half] = 1
        df2 = cudf.DataFrame()
        df2['in'] = random_array
        df2['indicator'] = indicator

        pdf_low = pd.DataFrame()
        pdf_high = pd.DataFrame()
        pdf_low['in'] = random_array[0:half]
        pdf_high['in'] = random_array[half:]

        self._cudf_data_m = df2
        self._plow_data = pdf_low
        self._phigh_data = pdf_high

    def tearDown(self):
        pass

    @ordered
    def test_fractional_diff(self):
        '''Test frac diff method'''
        for d_val in [0.1, 0.5, 1.0]:
            for floor_val in [1e-3, 1e-4]:
                gres, weights = fractional_diff(self._cudf_data['in'], d=d_val,
                                                floor=floor_val)
                pres, weights = frac_diff(self._pandas_data, d=d_val,
                                          floor=floor_val)
                length = weights.size
                g_array = (np.array(gres)[length-1:-1])
                p_array = (pres[0].values)
                err = abs(g_array - p_array).max()
                msg = "bad error %f\n" % (err,)
                self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_multi_fractional_diff(self):
        '''Test frac diff method'''
        d_val = 0.5
        floor_val = 1e-3
        gres, weights = port_fractional_diff(self._cudf_data_m['indicator'],
                                             self._cudf_data_m['in'], d=d_val,
                                             floor=floor_val)
        pres, weights = frac_diff(self._plow_data, d=d_val,
                                  floor=floor_val)
        length = weights.size
        g_array = (np.array(gres)[length-1:self.half-1])
        # make sure nan is set at the begining
        self.assertTrue(np.isnan(np.array(gres)[:length-1]).all())
        p_array = (pres[0].values)
        err = abs(g_array - p_array).max()
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        pres, weights = frac_diff(self._phigh_data, d=d_val,
                                  floor=floor_val)
        length = weights.size
        g_array = (np.array(gres)[self.half+length-1:-1])
        # make sure nan is set at the begining
        self.assertTrue(np.isnan(
            np.array(gres)[self.half:self.half+length-1]).all())
        p_array = (pres[0].values)
        err = abs(g_array - p_array).max()
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)


if __name__ == '__main__':
    unittest.main()
