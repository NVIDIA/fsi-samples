'''
Workflow Serialization Unit Tests

To run unittests:

# Using standard library unittest

python -m unittest -v
python -m unittest tests/unit/test_util.py -v

or

python -m unittest discover <test_directory>
python -m unittest discover -s <directory> -p 'test_*.py'

# Using pytest
# "conda install pytest" or "pip install pytest"
pytest -v tests
pytest -v tests/unit/test_util.py

'''
import pandas as pd
import unittest
import cudf
from gquant.cuindicator import shift, diff
import numpy as np
from .utils import make_orderer, error_function

ordered, compare = make_orderer()
unittest.defaultTestLoader.sortTestMethodsUsing = compare


class TestUtil(unittest.TestCase):

    def setUp(self):
        array_len = int(1e4)
        self.average_window = 300
        random_array = np.random.rand(array_len)

        df = cudf.DataFrame()
        df['in'] = random_array

        pdf = pd.DataFrame()
        pdf['in'] = random_array

        # ignore importlib warnings.
        self._pandas_data = pdf
        self._cudf_data = df

    def tearDown(self):
        pass

    @ordered
    def test_diff_functions(self):
        '''Test diff method'''
        for window in [-1, -2, -3, 1, 2, 3]:
            gpu_result = diff(self._cudf_data['in'], window)
            cpu_result = self._pandas_data['in'].diff(window)
            err = error_function(cudf.Series(gpu_result, nan_as_null=False), cpu_result)
            msg = "bad error %f\n" % (err,)
            self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_shift_functions(self):
        '''Test shift method'''
        for window in [-1, -2, -3, 1, 2, 3]:
            gpu_result = shift(self._cudf_data['in'], window)
            cpu_result = self._pandas_data['in'].shift(window)
            err = error_function(cudf.Series(gpu_result, nan_as_null=False), cpu_result)
            msg = "bad error %f\n" % (err,)
            self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)


if __name__ == '__main__':
    unittest.main()
