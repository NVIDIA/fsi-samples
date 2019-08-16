'''
Technical Indicator for Multiple Assets Unit Tests

To run unittests:

# Using standard library unittest

python -m unittest -v
python -m unittest tests/unit/test_multi_assets_indicator.py -v

or

python -m unittest discover <test_directory>
python -m unittest discover -s <directory> -p 'test_*.py'

# Using pytest
# "conda install pytest" or "pip install pytest"
pytest -v tests
pytest -v tests/unit/test_multi_assets_indicator.py

'''
import pandas as pd
import unittest
import cudf
from .utils import make_orderer, error_function
import gquant.cuindicator as gi
from . import technical_indicators as ti
from gquant.cuindicator import PEwm
import numpy as np
import warnings

ordered, compare = make_orderer()
unittest.defaultTestLoader.sortTestMethodsUsing = compare


class TestMultipleAssets(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings('ignore', message='numpy.ufunc size changed')
        warnings.simplefilter('ignore', category=ImportWarning)
        warnings.simplefilter('ignore', category=DeprecationWarning)
        size = 200
        half = size // 2
        self.size = size
        self.half = half
        np.random.seed(10)
        random_array = np.random.rand(size)
        open_array = np.random.rand(size)
        close_array = np.random.rand(size)
        high_array = np.random.rand(size)
        low_array = np.random.rand(size)
        volume_array = np.random.rand(size)
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
        low_pdf['High'] = high_array[0:half]
        low_pdf['Low'] = low_array[0:half]
        low_pdf['Volume'] = volume_array[0:half]

        high_pdf['Open'] = open_array[half:]
        high_pdf['Close'] = close_array[half:]
        high_pdf['High'] = high_array[half:]
        high_pdf['Low'] = low_array[half:]
        high_pdf['Volume'] = volume_array[half:]

        self._pandas_data = pdf
        self._cudf_data = df
        self._plow_data = low_pdf
        self._phigh_data = high_pdf

    def tearDown(self):
        pass

    @ordered
    def test_multi_assets_indicator(self):
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
        r = gi.port_ppsr(self._cudf_data['indicator'],
                         self._cudf_data['high'],
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

    @ordered
    def test_port_stochastic_oscillator_k(self):
        '''Test portfolio stochastic oscillator'''
        r = gi.port_stochastic_oscillator_k(self._cudf_data['indicator'],
                                            self._cudf_data['high'],
                                            self._cudf_data['low'],
                                            self._cudf_data['close'])

        cpu_result = ti.stochastic_oscillator_k(self._plow_data)
        err = error_function(r[:self.half], cpu_result['SO%k'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        cpu_result = ti.stochastic_oscillator_k(self._phigh_data)
        err = error_function(r[self.half:], cpu_result['SO%k'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_port_stochastic_oscillator_d(self):
        '''Test portfolio stochastic oscillator'''
        n = 10
        r = gi.port_stochastic_oscillator_d(self._cudf_data['indicator'],
                                            self._cudf_data['high'],
                                            self._cudf_data['low'],
                                            self._cudf_data['close'],
                                            n)

        cpu_result = ti.stochastic_oscillator_d(self._plow_data, n)
        err = error_function(r[:self.half], cpu_result['SO%d_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        cpu_result = ti.stochastic_oscillator_d(self._phigh_data, n)
        err = error_function(r[self.half:], cpu_result['SO%d_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_port_moving_average(self):
        '''Test portfolio moving average'''
        n = 10
        r = gi.port_moving_average(self._cudf_data['indicator'],
                                   self._cudf_data['close'],
                                   n)

        cpu_result = ti.moving_average(self._plow_data, n)
        err = error_function(r[:self.half], cpu_result['MA_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        cpu_result = ti.moving_average(self._phigh_data, n)
        err = error_function(r[self.half:], cpu_result['MA_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_port_rate_of_change(self):
        '''Test portfolio rate_of_change'''
        n = 10
        r = gi.port_rate_of_change(self._cudf_data['indicator'],
                                   self._cudf_data['close'],
                                   n)

        cpu_result = ti.rate_of_change(self._plow_data, n)
        err = error_function(r[:self.half], cpu_result['ROC_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        cpu_result = ti.rate_of_change(self._phigh_data, n)
        err = error_function(r[self.half:], cpu_result['ROC_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        n = -10
        r = gi.port_rate_of_change(self._cudf_data['indicator'],
                                   self._cudf_data['close'],
                                   n)

        cpu_result = ti.rate_of_change(self._plow_data, n)
        err = error_function(r[:self.half], cpu_result['ROC_-10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        cpu_result = ti.rate_of_change(self._phigh_data, n)
        err = error_function(r[self.half:], cpu_result['ROC_-10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_port_diff(self):
        '''Test portfolio diff'''
        n = 10
        r = gi.port_diff(self._cudf_data['indicator'],
                         self._cudf_data['close'],
                         n)

        cpu_result = self._plow_data['Close'].diff(n)
        err = error_function(r[:self.half], cpu_result)
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        cpu_result = self._phigh_data['Close'].diff(n)
        err = error_function(r[self.half:], cpu_result)
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        n = -10
        r = gi.port_diff(self._cudf_data['indicator'],
                         self._cudf_data['close'],
                         n)

        cpu_result = self._plow_data['Close'].diff(n)
        err = error_function(r[:self.half], cpu_result)
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        cpu_result = self._phigh_data['Close'].diff(n)
        err = error_function(r[self.half:], cpu_result)
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_port_shift(self):
        '''Test portfolio shift'''
        n = 10
        r = gi.port_shift(self._cudf_data['indicator'],
                          self._cudf_data['close'],
                          n)

        cpu_result = self._plow_data['Close'].shift(n)
        err = error_function(r[:self.half], cpu_result)
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        cpu_result = self._phigh_data['Close'].shift(n)
        err = error_function(r[self.half:], cpu_result)
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        n = -10
        r = gi.port_shift(self._cudf_data['indicator'],
                          self._cudf_data['close'],
                          n)

        cpu_result = self._plow_data['Close'].shift(n)
        err = error_function(r[:self.half], cpu_result)
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        cpu_result = self._phigh_data['Close'].shift(n)
        err = error_function(r[self.half:], cpu_result)
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_port_bollinger_bands(self):
        '''Test portfolio bollinger bands'''
        n = 10
        r = gi.port_bollinger_bands(self._cudf_data['indicator'],
                                    self._cudf_data['close'],
                                    n)

        cpu_result = ti.bollinger_bands(self._plow_data, n)
        err = error_function(r.b1[:self.half], cpu_result['BollingerB_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        err = error_function(r.b2[:self.half], cpu_result['Bollinger%b_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        cpu_result = ti.bollinger_bands(self._phigh_data, n)
        err = error_function(r.b1[self.half:], cpu_result['BollingerB_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        err = error_function(r.b2[self.half:], cpu_result['Bollinger%b_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_port_average_directional_movement_index(self):
        '''Test portfolio average directional movement index'''
        n = 10
        n_adx = 20
        r = gi.port_average_directional_movement_index(
            self._cudf_data['indicator'],
            self._cudf_data['high'],
            self._cudf_data['low'],
            self._cudf_data['close'],
            n, n_adx)

        cpu_result = ti.average_directional_movement_index(self._plow_data,
                                                           n,
                                                           n_adx)
        err = error_function(r[:self.half], cpu_result['ADX_10_20'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)
        cpu_result = ti.average_directional_movement_index(self._phigh_data,
                                                           n,
                                                           n_adx)
        err = error_function(r[self.half:], cpu_result['ADX_10_20'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_port_vortex_indicator(self):
        '''Test portfolio vortex indicator'''
        n = 10
        r = gi.port_vortex_indicator(
            self._cudf_data['indicator'],
            self._cudf_data['high'],
            self._cudf_data['low'],
            self._cudf_data['close'],
            n)

        cpu_result = ti.vortex_indicator(self._plow_data,
                                         n)
        err = error_function(r[:self.half], cpu_result['Vortex_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)
        cpu_result = ti.vortex_indicator(self._phigh_data,
                                         n)
        err = error_function(r[self.half:], cpu_result['Vortex_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_port_kst_oscillator(self):
        '''Test portfolio kst oscillator'''

        r = gi.port_kst_oscillator(
            self._cudf_data['indicator'],
            self._cudf_data['close'], 3, 4, 5, 6, 7, 8, 9, 10)

        cpu_result = ti.kst_oscillator(self._plow_data,
                                       3, 4, 5, 6, 7, 8, 9, 10)
        err = error_function(r[:self.half], cpu_result['KST_3_4_5_6_7_8_9_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)
        cpu_result = ti.kst_oscillator(self._phigh_data,
                                       3, 4, 5, 6, 7, 8, 9, 10)
        err = error_function(r[self.half:], cpu_result['KST_3_4_5_6_7_8_9_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_port_mass_index(self):
        '''Test portfolio mass index'''

        r = gi.port_mass_index(
            self._cudf_data['indicator'],
            self._cudf_data['high'],
            self._cudf_data['low'],
            9, 25)

        cpu_result = ti.mass_index(self._plow_data)
        err = error_function(r[:self.half], cpu_result['Mass Index'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)
        cpu_result = ti.mass_index(self._phigh_data)
        err = error_function(r[self.half:], cpu_result['Mass Index'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_port_true_strength_index(self):
        '''Test portfolio true strength index'''

        r = gi.port_true_strength_index(
            self._cudf_data['indicator'],
            self._cudf_data['close'],
            5, 8)

        cpu_result = ti.true_strength_index(self._plow_data, 5, 8)
        err = error_function(r[:self.half], cpu_result['TSI_5_8'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)
        cpu_result = ti.true_strength_index(self._phigh_data, 5, 8)
        err = error_function(r[self.half:], cpu_result['TSI_5_8'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_port_chaikin_oscillator(self):
        '''Test portfolio chaikin oscillator'''

        r = gi.port_chaikin_oscillator(
            self._cudf_data['indicator'],
            self._cudf_data['high'],
            self._cudf_data['low'],
            self._cudf_data['close'],
            self._cudf_data['volume'],
            3, 10)

        cpu_result = ti.chaikin_oscillator(self._plow_data)
        err = error_function(r[:self.half], cpu_result['Chaikin'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)
        cpu_result = ti.chaikin_oscillator(self._phigh_data)
        err = error_function(r[self.half:], cpu_result['Chaikin'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_port_money_flow_index(self):
        '''Test portfolio money flow index'''

        r = gi.port_money_flow_index(
            self._cudf_data['indicator'],
            self._cudf_data['high'],
            self._cudf_data['low'],
            self._cudf_data['close'],
            self._cudf_data['volume'],
            10)

        cpu_result = ti.money_flow_index(self._plow_data, 10)
        err = error_function(r[:self.half], cpu_result['MFI_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)
        cpu_result = ti.money_flow_index(self._phigh_data, 10)
        err = error_function(r[self.half:], cpu_result['MFI_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_port_on_balance_volume(self):
        '''Test portfolio on balance volume'''

        r = gi.port_on_balance_volume(
            self._cudf_data['indicator'],
            self._cudf_data['close'],
            self._cudf_data['volume'],
            10)

        cpu_result = ti.on_balance_volume(self._plow_data, 10)
        err = error_function(r[:self.half], cpu_result['OBV_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)
        cpu_result = ti.on_balance_volume(self._phigh_data, 10)
        err = error_function(r[self.half:], cpu_result['OBV_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_port_force_index(self):
        '''Test portfolio force index'''

        r = gi.port_force_index(
            self._cudf_data['indicator'],
            self._cudf_data['close'],
            self._cudf_data['volume'],
            10)

        cpu_result = ti.force_index(self._plow_data, 10)
        err = error_function(r[:self.half], cpu_result['Force_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)
        cpu_result = ti.force_index(self._phigh_data, 10)
        err = error_function(r[self.half:], cpu_result['Force_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_port_ease_of_movement(self):
        '''Test portfolio ease of movement'''

        r = gi.port_ease_of_movement(
            self._cudf_data['indicator'],
            self._cudf_data['high'],
            self._cudf_data['low'],
            self._cudf_data['volume'],
            10)

        cpu_result = ti.ease_of_movement(self._plow_data, 10)
        err = error_function(r[:self.half], cpu_result['EoM_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)
        cpu_result = ti.ease_of_movement(self._phigh_data, 10)
        err = error_function(r[self.half:], cpu_result['EoM_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_port_ultimate_oscillator(self):
        '''Test portfolio ultimate oscillator'''

        r = gi.port_ultimate_oscillator(
            self._cudf_data['indicator'],
            self._cudf_data['high'],
            self._cudf_data['low'],
            self._cudf_data['close'])

        cpu_result = ti.ultimate_oscillator(self._plow_data)
        err = error_function(r[:self.half], cpu_result['Ultimate_Osc'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)
        cpu_result = ti.ultimate_oscillator(self._phigh_data)
        err = error_function(r[self.half:], cpu_result['Ultimate_Osc'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_port_donchian_channel(self):
        '''Test portfolio donchian channel'''

        r = gi.port_donchian_channel(
            self._cudf_data['indicator'],
            self._cudf_data['high'],
            self._cudf_data['low'],
            10)
        cpu_result = ti.donchian_channel(self._plow_data, 10)
        err = error_function(r[:self.half-1], cpu_result['Donchian_10'][0:99])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)
        cpu_result = ti.donchian_channel(self._phigh_data, 10)
        err = error_function(r[self.half:-1], cpu_result['Donchian_10'][0:99])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_port_coppock_curve(self):
        '''Test portfolio coppock curve'''

        r = gi.port_coppock_curve(
            self._cudf_data['indicator'],
            self._cudf_data['close'],
            10)
        cpu_result = ti.coppock_curve(self._plow_data, 10)
        err = error_function(r[:self.half], cpu_result['Copp_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)
        cpu_result = ti.coppock_curve(self._phigh_data, 10)
        err = error_function(r[self.half:], cpu_result['Copp_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_port_accumulation_distribution(self):
        '''Test portfolio accumulation distribution'''

        r = gi.port_accumulation_distribution(
            self._cudf_data['indicator'],
            self._cudf_data['high'],
            self._cudf_data['low'],
            self._cudf_data['close'],
            self._cudf_data['volume'],
            10)
        cpu_result = ti.accumulation_distribution(self._plow_data, 10)
        err = error_function(r[:self.half], cpu_result['Acc/Dist_ROC_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)
        cpu_result = ti.accumulation_distribution(self._phigh_data, 10)
        err = error_function(r[self.half:], cpu_result['Acc/Dist_ROC_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_port_commodity_channel_index(self):
        '''Test portfolio commodity channel index'''

        r = gi.port_commodity_channel_index(
            self._cudf_data['indicator'],
            self._cudf_data['high'],
            self._cudf_data['low'],
            self._cudf_data['close'],
            10)
        cpu_result = ti.commodity_channel_index(self._plow_data, 10)
        err = error_function(r[:self.half], cpu_result['CCI_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)
        cpu_result = ti.commodity_channel_index(self._phigh_data, 10)
        err = error_function(r[self.half:], cpu_result['CCI_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_port_keltner_channel(self):
        '''Test portfolio keltner channel'''

        r = gi.port_keltner_channel(
            self._cudf_data['indicator'],
            self._cudf_data['high'],
            self._cudf_data['low'],
            self._cudf_data['close'],
            10)
        cpu_result = ti.keltner_channel(self._plow_data, 10)
        err = error_function(r.KelChD[:self.half], cpu_result['KelChD_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        err = error_function(r.KelChM[:self.half], cpu_result['KelChM_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        err = error_function(r.KelChU[:self.half], cpu_result['KelChU_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        cpu_result = ti.keltner_channel(self._phigh_data, 10)
        err = error_function(r.KelChD[self.half:], cpu_result['KelChD_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        err = error_function(r.KelChM[self.half:], cpu_result['KelChM_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        err = error_function(r.KelChU[self.half:], cpu_result['KelChU_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)


if __name__ == '__main__':
    unittest.main()
