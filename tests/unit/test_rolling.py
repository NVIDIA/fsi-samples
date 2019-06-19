'''
Workflow Serialization Unit Tests

To run unittests:

# Using standard library unittest

python -m unittest -v
python -m unittest tests/unit/test_rolling.py -v

or

python -m unittest discover <test_directory>
python -m unittest discover -s <directory> -p 'test_*.py'

# Using pytest
# "conda install pytest" or "pip install pytest"
pytest -v tests
pytest -v tests/unit/test_rolling.py

'''
import os
import warnings
from io import StringIO
import shutil, tempfile
import pandas as pd
import unittest
import pathlib
import cudf
from gquant.cuindicator import Rolling, Ewm
import numpy as np
# --------------------------------------------------------- Keep tests in order
def make_orderer():
    order = {}

    def ordered(f):
        order[f.__name__] = len(order)
        return f

    def compare(a, b):
        return [1, -1][order[a] < order[b]]

    return ordered, compare

ordered, compare = make_orderer()
unittest.defaultTestLoader.sortTestMethodsUsing = compare

# ------------------------------------------- Workflow Serialization Test Cases

def error_function(gpu_arr, result_series):
    gpu_arr = gpu_arr.to_array(fillna='pandas')
    pan_arr = result_series.values
    gpu_arr = gpu_arr[~np.isnan(gpu_arr) & ~np.isinf(gpu_arr)]
    pan_arr = pan_arr[~np.isnan(pan_arr) & ~np.isinf(pan_arr)]
    err = np.abs(gpu_arr - pan_arr).max()
    return err

class TestIndicator(unittest.TestCase):

    def setUp(self):
        array_len = int(1e4)
        self.average_window = 300
        number_type = np.float64
        random_array = np.random.rand(array_len)

        df = cudf.dataframe.DataFrame()
        df['in'] = random_array

        pdf = pd.DataFrame()
        pdf['in'] = random_array

        # ignore importlib warnings.
        self._pandas_data = pdf
        self._cudf_data = df

    def tearDown(self):
        pass

    @ordered
    def test_rolling_functions(self):
        '''Test rolling window method'''

        gpu_result = Rolling(self.average_window, self._cudf_data['in']).mean()
        cpu_result = self._pandas_data['in'].rolling(self.average_window).mean()
        err = error_function(cudf.Series(gpu_result), cpu_result)
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        gpu_result = Rolling(self.average_window, self._cudf_data['in']).max()
        cpu_result = self._pandas_data['in'].rolling(self.average_window).max()
        err = error_function(cudf.Series(gpu_result), cpu_result)
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        gpu_result = Rolling(self.average_window, self._cudf_data['in']).min()
        cpu_result = self._pandas_data['in'].rolling(self.average_window).min()
        err = error_function(cudf.Series(gpu_result), cpu_result)
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        gpu_result = Rolling(self.average_window, self._cudf_data['in']).sum()
        cpu_result = self._pandas_data['in'].rolling(self.average_window).sum()
        err = error_function(cudf.Series(gpu_result), cpu_result)
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        gpu_result = Rolling(self.average_window, self._cudf_data['in']).std()
        cpu_result = self._pandas_data['in'].rolling(self.average_window).std()
        err = error_function(cudf.Series(gpu_result), cpu_result)
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        gpu_result = Rolling(self.average_window, self._cudf_data['in']).var()
        cpu_result = self._pandas_data['in'].rolling(self.average_window).var()
        err = error_function(cudf.Series(gpu_result), cpu_result)
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_ewm_functions(self):
        '''Test exponential moving average method'''
        gpu_result = Ewm(self.average_window, self._cudf_data['in']).mean()
        cpu_result = self._pandas_data[
            'in'].ewm(span=self.average_window,
                      min_periods=self.average_window).mean()
        err = error_function(cudf.Series(gpu_result), cpu_result)
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)


if __name__ == '__main__':
    unittest.main()
