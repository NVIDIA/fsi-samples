'''
Workflow Serialization Unit Tests

To run unittests:

# Using standard library unittest

python -m unittest -v
python -m unittest tests/unit/test_indicator.py -v

or

python -m unittest discover <test_directory>
python -m unittest discover -s <directory> -p 'test_*.py'

# Using pytest
# "conda install pytest" or "pip install pytest"
pytest -v tests
pytest -v tests/unit/test_indicator.py

'''
import os
import warnings
from io import StringIO
import shutil, tempfile
import pandas as pd
import unittest
import pathlib
import cudf
from difflib import context_diff
from gquant.cuindicator import (rate_of_change, bollinger_bands, trix, macd,
                                average_true_range, ppsr,
                                stochastic_oscillator_k,
                                stochastic_oscillator_d,
                                average_directional_movement_index,
                                vortex_indicator,
                                kst_oscillator, relative_strength_index,
                                mass_index, true_strength_index,
                                chaikin_oscillator,
                                money_flow_index, on_balance_volume,
                                force_index, ease_of_movement,
                                ultimate_oscillator, donchian_channel,
                                keltner_channel, coppock_curve,
                                accumulation_distribution,
                                commodity_channel_index, moving_average,
                                exponential_moving_average, momentum)

from . import technical_indicators as p
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
        # ignore importlib warnings.
        path = pathlib.Path(__file__)
        self._pandas_data = pd.read_csv(str(path.parent)+'/testdata.csv.gz')
        self._pandas_data['Volume'] /= 1000.0
        self._cudf_data = cudf.from_pandas(self._pandas_data)
        warnings.simplefilter('ignore', category=ImportWarning)
        warnings.simplefilter('ignore', category=DeprecationWarning)


    def tearDown(self):
        pass

    @ordered
    def test_rate_of_return(self):
        '''Test rate of return calculation'''
        r_cudf = rate_of_change(self._cudf_data['Close'], 2)
        r_pandas = p.rate_of_change(self._pandas_data, 2)
        err = error_function(r_cudf, r_pandas.ROC_2)
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_trix(self):
        """ test the trix calculation"""
        r_cudf = trix(self._cudf_data['Close'], 3)
        r_pandas = p.trix(self._pandas_data, 3)
        err = error_function(r_cudf, r_pandas.Trix_3)
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_bollinger_bands(self):
        """ test the bollinger_bands """
        r_cudf = bollinger_bands(self._cudf_data['Close'], 20)
        r_pandas = p.bollinger_bands(self._pandas_data, 20)
        err = error_function(r_cudf.b1, r_pandas['BollingerB_20'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)
        err = error_function(r_cudf.b2, r_pandas['Bollinger%b_20'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_macd(self):
        """ test the macd """
        n_fast = 10
        n_slow = 20
        r_cudf = macd(self._cudf_data['Close'], n_fast, n_slow)
        r_pandas = p.macd(self._pandas_data, n_fast, n_slow)
        err = error_function(r_cudf.MACD, r_pandas['MACD_10_20'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)
        err = error_function(r_cudf.MACDdiff, r_pandas['MACDdiff_10_20'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)
        err = error_function(r_cudf.MACDsign, r_pandas['MACDsign_10_20'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_average_true_range(self):
        """ test the average true range """
        r_cudf = average_true_range(self._cudf_data['High'],
                                    self._cudf_data['Low'],
                                    self._cudf_data['Close'], 10)
        r_pandas = p.average_true_range(self._pandas_data, 10)
        err = error_function(r_cudf, r_pandas['ATR_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_average_true_range(self):
        """ test the average true range """
        r_cudf = ppsr(self._cudf_data['High'], self._cudf_data['Low'],
                      self._cudf_data['Close'])
        r_pandas = p.ppsr(self._pandas_data)
        err = error_function(r_cudf.PP, r_pandas['PP'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)
        err = error_function(r_cudf.R1, r_pandas['R1'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)
        err = error_function(r_cudf.S1, r_pandas['S1'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)
        err = error_function(r_cudf.R2, r_pandas['R2'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)
        err = error_function(r_cudf.S2, r_pandas['S2'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)
        err = error_function(r_cudf.R3, r_pandas['R3'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)
        err = error_function(r_cudf.S3, r_pandas['S3'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)


    @ordered
    def test_stochastic_oscillator_k(self):
        """ test the stochastic oscillator k """
        r_cudf = stochastic_oscillator_k(self._cudf_data['High'],
                                         self._cudf_data['Low'],
                                         self._cudf_data['Close'])
        r_pandas = p.stochastic_oscillator_k(self._pandas_data)
        err = error_function(r_cudf, r_pandas['SO%k'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_stochastic_oscillator_d(self):
        """ test the stochastic oscillator d """
        r_cudf = stochastic_oscillator_d(self._cudf_data['High'],
                                         self._cudf_data['Low'],
                                         self._cudf_data['Close'], 10)
        r_pandas = p.stochastic_oscillator_d(self._pandas_data, 10)
        err = error_function(r_cudf, r_pandas['SO%d_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_average_directional_movement_index(self):
        """ test the average_directional_movement_index """
        r_cudf = average_directional_movement_index(self._cudf_data['High'],
                                                    self._cudf_data['Low'],
                                                    self._cudf_data['Close'],
                                                    10, 20)
        r_pandas = p.average_directional_movement_index(self._pandas_data,
                                                        10, 20)
        err = error_function(r_cudf, r_pandas['ADX_10_20'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_vortex_indicator(self):
        """ test the vortex_indicator """
        r_cudf = vortex_indicator(self._cudf_data['High'],
                                  self._cudf_data['Low'],
                                  self._cudf_data['Close'], 10)
        r_pandas = p.vortex_indicator(self._pandas_data, 10)
        err = error_function(r_cudf, r_pandas['Vortex_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_kst_oscillator(self):
        """ test the kst_oscillator """
        r_cudf = kst_oscillator(self._cudf_data['Close'],
                                3, 4, 5, 6, 7, 8, 9, 10)
        r_pandas = p.kst_oscillator(self._pandas_data,
                                    3, 4, 5, 6, 7, 8, 9, 10)
        err = error_function(r_cudf, r_pandas['KST_3_4_5_6_7_8_9_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_relative_strength_index(self):
        """ test the relative_strength_index """
        r_cudf = relative_strength_index(self._cudf_data['High'],
                                         self._cudf_data['Low'], 10)
        r_pandas = p.relative_strength_index(self._pandas_data, 10)
        err = error_function(r_cudf, r_pandas['RSI_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)
    @ordered
    def test_mass_index(self):
        """ test the mass_index """
        r_cudf = mass_index(self._cudf_data['High'],
                            self._cudf_data['Low'], 9, 25)
        r_pandas = p.mass_index(self._pandas_data)
        err = error_function(r_cudf, r_pandas['Mass Index'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)
    @ordered
    def test_true_strength_index(self):
        """ test the true_strength_index """
        r_cudf = true_strength_index(self._cudf_data['Close'], 5, 8)
        r_pandas = p.true_strength_index(self._pandas_data, 5, 8)
        err = error_function(r_cudf, r_pandas['TSI_5_8'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_chaikin_oscillator(self):
        """ test the chaikin_oscillator """
        r_cudf = chaikin_oscillator(self._cudf_data['High'],
                                    self._cudf_data['Low'],
                                    self._cudf_data['Close'],
                                    self._cudf_data['Volume'],  3, 10)
        r_pandas = p.chaikin_oscillator(self._pandas_data)
        err = error_function(r_cudf, r_pandas['Chaikin'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_money_flow_index(self):
        """ test the money_flow_index """
        r_cudf = money_flow_index(self._cudf_data['High'],
                                  self._cudf_data['Low'],
                                  self._cudf_data['Close'],
                                  self._cudf_data['Volume'], 10)
        r_pandas = p.money_flow_index(self._pandas_data, 10)
        err = error_function(r_cudf, r_pandas['MFI_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_on_balance_volume(self):
        """ test the on_balance_volume """
        r_cudf = on_balance_volume(self._cudf_data['Close'],
                                   self._cudf_data['Volume'], 10)
        r_pandas = p.on_balance_volume(self._pandas_data, 10)
        err = error_function(r_cudf, r_pandas['OBV_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_money_flow_index(self):
        """ test the money_flow_index """
        r_cudf = force_index(self._cudf_data['Close'],
                             self._cudf_data['Volume'], 10)
        r_pandas = p.force_index(self._pandas_data, 10)
        err = error_function(r_cudf, r_pandas['Force_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_ease_of_movement(self):
        """ test the ease_of_movement """
        r_cudf = ease_of_movement(self._cudf_data['High'],
                                  self._cudf_data['Low'],
                                  self._cudf_data['Volume'], 10)
        r_pandas = p.ease_of_movement(self._pandas_data, 10)
        err = error_function(r_cudf, r_pandas['EoM_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_ultimate_oscillator(self):
        """ test the ultimate_oscillator """
        r_cudf = ultimate_oscillator(self._cudf_data['High'],
                                     self._cudf_data['Low'],
                                     self._cudf_data['Close'])
        r_pandas = p.ultimate_oscillator(self._pandas_data)
        err = error_function(r_cudf, r_pandas['Ultimate_Osc'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_donchian_channel(self):
        """ test the donchian_channel """
        r_cudf = donchian_channel(self._cudf_data['High'],
                                  self._cudf_data['Low'], 10)
        r_pandas = p.donchian_channel(self._pandas_data, 10)
        err = error_function(r_cudf[:-1], r_pandas['Donchian_10'][:-1])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_keltner_channel(self):
        """ test the keltner_channel """
        r_cudf = keltner_channel(self._cudf_data['High'],
                                 self._cudf_data['Low'],
                                 self._cudf_data['Close'], 10)
        r_pandas = p.keltner_channel(self._pandas_data, 10)
        err = error_function(r_cudf.KelChD, r_pandas['KelChD_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)
        err = error_function(r_cudf.KelChM, r_pandas['KelChM_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)
        err = error_function(r_cudf.KelChU, r_pandas['KelChU_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_coppock_curve(self):
        """ test the coppock_curve """
        r_cudf = coppock_curve(self._cudf_data['Close'], 10)
        r_pandas = p.coppock_curve(self._pandas_data, 10)
        err = error_function(r_cudf, r_pandas['Copp_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_accumulation_distribution(self):
        """ test the accumulation_distribution """
        r_cudf = accumulation_distribution(self._cudf_data['High'],
                                           self._cudf_data['Low'],
                                           self._cudf_data['Close'],
                                           self._cudf_data['Volume'], 10)
        r_pandas = p.accumulation_distribution(self._pandas_data, 10)
        err = error_function(r_cudf, r_pandas['Acc/Dist_ROC_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_commodity_channel_index(self):
        """ test the commodity_channel_index """
        r_cudf = commodity_channel_index(self._cudf_data['High'],
                                         self._cudf_data['Low'],
                                         self._cudf_data['Close'], 10)
        r_pandas = p.commodity_channel_index(self._pandas_data, 10)
        err = error_function(r_cudf, r_pandas['CCI_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_momentum(self):
        """ test the momentum """
        r_cudf = momentum(self._cudf_data['Close'], 10)
        r_pandas = p.momentum(self._pandas_data, 10)
        err = error_function(r_cudf, r_pandas['Momentum_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_moving_average(self):
        """ test the moving average """
        r_cudf = moving_average(self._cudf_data['Close'], 10)
        r_pandas = p.moving_average(self._pandas_data, 10)
        err = error_function(r_cudf, r_pandas['MA_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

    @ordered
    def test_exponential_moving_average(self):
        """ test the exponential moving average """
        r_cudf = exponential_moving_average(self._cudf_data['Close'], 10)
        r_pandas = p.exponential_moving_average(self._pandas_data, 10)
        err = error_function(r_cudf, r_pandas['EMA_10'])
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

if __name__ == '__main__':
    unittest.main()
