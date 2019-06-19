'''
Workflow Serialization Unit Tests

To run unittests:

# Using standard library unittest

python -m unittest -v
python -m unittest tests/unit/test_pewm.py -v

or

python -m unittest discover <test_directory>
python -m unittest discover -s <directory> -p 'test_*.py'

# Using pytest
# "conda install pytest" or "pip install pytest"
pytest -v tests
pytest -v tests/unit/test_pewm.py

'''
import pandas as pd
import unittest
import cudf
from .utils import make_orderer, error_function
from gquant.cuindicator import PEwm
import numpy as np

ordered, compare = make_orderer()
unittest.defaultTestLoader.sortTestMethodsUsing = compare


class TestIndicator(unittest.TestCase):

    def setUp(self):
        random_array = np.arange(20, dtype=np.float64)
        indicator = np.zeros(20, dtype=np.int32)
        indicator[0] = 1
        indicator[10] = 1
        df = cudf.dataframe.DataFrame()
        df['in'] = random_array
        df['indicator'] = indicator

        pdf = pd.DataFrame()
        pdf['in0'] = random_array[0:10]
        pdf['in1'] = random_array[10:]

        # ignore importlib warnings.
        self._pandas_data = pdf
        self._cudf_data = df

    def tearDown(self):
        pass

    @ordered
    def test_pewm(self):
        '''Test portfolio ewm method'''
        self._cudf_data['ewma'] = PEwm(3,
                                       self._cudf_data['in'],
                                       self._cudf_data[
                                           'indicator'].data.to_gpu_array(),
                                       thread_tile=2,
                                       number_of_threads=2).mean()
        gpu_array = self._cudf_data['ewma']
        gpu_result = gpu_array[0:10]
        cpu_result = self._pandas_data['in0'].ewm(span=3,
                                                  min_periods=3).mean()
        err = error_function(gpu_result, cpu_result)
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        cpu_result = self._pandas_data['in1'].ewm(span=3,
                                                  min_periods=3).mean()
        gpu_result = gpu_array[10:20]
        err = error_function(gpu_result, cpu_result)
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)


if __name__ == '__main__':
    unittest.main()
