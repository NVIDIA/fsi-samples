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
import gquant.cuindicator as gi
from . import technical_indicators as ti
from gquant.cuindicator import PEwm
import numpy as np

ordered, compare = make_orderer()
unittest.defaultTestLoader.sortTestMethodsUsing = compare


class TestPEwm(unittest.TestCase):

    def setUp(self):
        size = 200
        half = size // 2
        self.size = size
        self.half = half
        random_array = np.arange(size, dtype=np.float64)
        open_array = np.arange(size, dtype=np.float64)
        close_array = np.arange(size, dtype=np.float64)
        high_array = np.arange(size, dtype=np.float64)
        low_array = np.arange(size, dtype=np.float64)
        volume_array = np.arange(size, dtype=np.float64)
        indicator = np.zeros(size, dtype=np.int32)
        indicator[0] = 1
        indicator[half] = 1
        df = cudf.dataframe.DataFrame()
        df['in'] = random_array
        df['open'] = open_array
        df['close'] = close_array
        df['high'] = high_array
        df['low'] = low_array
        df['volume'] = volume_array
        df['indicator'] = indicator

        pdf = pd.DataFrame()
        pdf['in0'] = random_array[0:half]
        pdf['in1'] = random_array[half:]

        low_pdf = pd.DataFrame()
        high_pdf = pd.DataFrame()

        low_pdf['Open'] = open_array[0:half]
        low_pdf['Close'] = close_array[0:half]
        low_pdf['High'] = open_array[0:half]
        low_pdf['Low'] = close_array[0:half]
        low_pdf['Volume'] = volume_array[0:half]

        high_pdf['Open'] = open_array[half:]
        high_pdf['Close'] = close_array[half:]
        high_pdf['High'] = open_array[half:]
        high_pdf['Low'] = close_array[half:]
        high_pdf['Volume'] = volume_array[half:]

        # ignore importlib warnings.
        self._pandas_data = pdf
        self._cudf_data = df
        self._plow_data = low_pdf
        self._phigh_data = high_pdf

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
        gpu_result = gpu_array[0:self.half]
        cpu_result = self._pandas_data['in0'].ewm(span=3,
                                                  min_periods=3).mean()
        err = error_function(gpu_result, cpu_result)
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        cpu_result = self._pandas_data['in1'].ewm(span=3,
                                                  min_periods=3).mean()
        gpu_result = gpu_array[self.half:]
        err = error_function(gpu_result, cpu_result)
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_port_macd(self):
        '''Test portfolio macd method'''
        n_fast = 10
        n_slow = 20
        r = gi.port_macd(self._cudf_data['indicator'].data.to_gpu_array(),
                         self._cudf_data['close'].data.to_gpu_array(),
                         n_fast,
                         n_slow)
        cpu_result = ti.macd(self._plow_data, n_fast, n_slow)
        err = error_function(r.MACD[:self.half], cpu_result['MACD_10_20'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)
        err = error_function(r.MACDsign[:self.half],
                             cpu_result['MACDsign_10_20'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)
        err = error_function(r.MACDdiff[:self.half],
                             cpu_result['MACDdiff_10_20'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        cpu_result = ti.macd(self._phigh_data, n_fast, n_slow)
        err = error_function(r.MACD[self.half:], cpu_result['MACD_10_20'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)
        err = error_function(r.MACDsign[self.half:],
                             cpu_result['MACDsign_10_20'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)
        err = error_function(r.MACDdiff[self.half:],
                             cpu_result['MACDdiff_10_20'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_port_relative_strength_index(self):
        '''Test portfolio relative strength index method'''
        n = 10
        r = gi.port_relative_strength_index(
            self._cudf_data['indicator'],
            self._cudf_data['high'],
            self._cudf_data['low'],
            n)

        cpu_result = ti.relative_strength_index(self._plow_data, n)
        err = error_function(r[:self.half], cpu_result['RSI_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        cpu_result = ti.relative_strength_index(self._phigh_data, n)
        err = error_function(r[self.half:], cpu_result['RSI_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_port_trix(self):
        '''Test portfolio trix'''
        n = 3

        r = gi.port_trix(self._cudf_data['indicator'],
                         self._cudf_data['close'],
                         n)

        cpu_result = ti.trix(self._plow_data, n)
        err = error_function(r[:self.half], cpu_result['Trix_3'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        cpu_result = ti.trix(self._phigh_data, n)
        err = error_function(r[self.half:], cpu_result['Trix_3'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_port_average_true_range(self):
        '''Test portfolio average true range'''
        n = 10
        r = gi.port_average_true_range(self._cudf_data['indicator'],
                                       self._cudf_data['high'],
                                       self._cudf_data['low'],
                                       self._cudf_data['close'], 10)

        cpu_result = ti.average_true_range(self._plow_data, n)
        err = error_function(r[:self.half], cpu_result['ATR_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        cpu_result = ti.average_true_range(self._phigh_data, n)
        err = error_function(r[self.half:], cpu_result['ATR_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_port_ppsr(self):
        '''Test portfolio average true range'''
        r = gi.ppsr(self._cudf_data['high'],
                    self._cudf_data['low'],
                    self._cudf_data['close'])

        cpu_result = ti.ppsr(self._plow_data)
        err = error_function(r.PP[:self.half], cpu_result['PP'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        err = error_function(r.R1[:self.half], cpu_result['R1'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        err = error_function(r.S1[:self.half], cpu_result['S1'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        err = error_function(r.R2[:self.half], cpu_result['R2'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        err = error_function(r.S2[:self.half], cpu_result['S2'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        err = error_function(r.R3[:self.half], cpu_result['R3'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        err = error_function(r.S3[:self.half], cpu_result['S3'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        cpu_result = ti.ppsr(self._phigh_data)
        err = error_function(r.PP[self.half:], cpu_result['PP'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        err = error_function(r.R1[self.half:], cpu_result['R1'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        err = error_function(r.S1[self.half:], cpu_result['S1'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        err = error_function(r.R2[self.half:], cpu_result['R2'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        err = error_function(r.S2[self.half:], cpu_result['S2'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        err = error_function(r.R3[self.half:], cpu_result['R3'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        err = error_function(r.S3[self.half:], cpu_result['S3'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)


if __name__ == '__main__':
    unittest.main()
