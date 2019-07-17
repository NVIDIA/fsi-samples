'''
Technical Indicator Node Unit Tests

To run unittests:

# Using standard library unittest

python -m unittest -v
python -m unittest tests/unit/test_indicator_node.py -v

or

python -m unittest discover <test_directory>
python -m unittest discover -s <directory> -p 'test_*.py'

# Using pytest
# "conda install pytest" or "pip install pytest"
pytest -v tests
pytest -v tests/unit/test_indicator_node.py

'''
import warnings
import unittest
import cudf
import gquant.cuindicator as gi
from gquant.plugin_nodes.transform.indicatorNode import IndicatorNode
from .utils import make_orderer
import numpy as np
import copy

ordered, compare = make_orderer()
unittest.defaultTestLoader.sortTestMethodsUsing = compare


class TestIndicatorNode(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)
        warnings.simplefilter('ignore', category=DeprecationWarning)
        # ignore importlib warnings.
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
        self._cudf_data = df
        self.conf = {
            "indicators": [
                {"function": "port_chaikin_oscillator",
                 "columns": ["high", "low", "close", "volume"],
                 "args": [10, 20]},
                {"function": "port_bollinger_bands",
                 "columns": ["close"],
                 "args": [10],
                 "outputs": ["b1", "b2"]}
            ],
            "remove_na": True
        }

    def tearDown(self):
        pass

    @ordered
    def test_colums(self):
        '''Test node columns requirments'''
        inN = IndicatorNode("abc", self.conf)

        col = "indicator"
        msg = "bad error: %s is missing" % (col)
        self.assertTrue(col in inN.required, msg)
        col = "high"
        msg = "bad error: %s is missing" % (col)
        self.assertTrue(col in inN.required, msg)
        col = "low"
        msg = "bad error: %s is missing" % (col)
        self.assertTrue(col in inN.required, msg)
        col = "close"
        msg = "bad error: %s is missing" % (col)
        self.assertTrue(col in inN.required, msg)
        col = "volume"
        msg = "bad error: %s is missing" % (col)
        self.assertTrue(col in inN.required, msg)

        col = "CH_OS_10_20"
        msg = "bad error: %s is missing" % (col)
        self.assertTrue(col in inN.addition, msg)
        col = "BO_BA_b1_10"
        msg = "bad error: %s is missing" % (col)
        self.assertTrue(col in inN.addition, msg)
        col = "BO_BA_b2_10"
        msg = "bad error: %s is missing" % (col)
        self.assertTrue(col in inN.addition, msg)

    @ordered
    def test_drop(self):
        '''Test node columns requirments'''
        inN = IndicatorNode("abc", self.conf)
        o = inN.process([self._cudf_data])
        msg = "bad error: df len %d is not right" % (len(o))
        self.assertTrue(len(o) == 162, msg)

        newConf = copy.deepcopy(self.conf)
        newConf['remove_na'] = False
        inN = IndicatorNode("abc", newConf)
        o = inN.process([self._cudf_data])
        msg = "bad error: df len %d is not right" % (len(o))
        self.assertTrue(len(o) == 200, msg)

    @ordered
    def test_signal(self):
        '''Test node columns requirments'''

        newConf = copy.deepcopy(self.conf)
        newConf['remove_na'] = False
        inN = IndicatorNode("abc", newConf)
        o = inN.process([self._cudf_data])
        # check chaikin oscillator computation
        r_cudf = gi.chaikin_oscillator(self._cudf_data[:self.half]['high'],
                                       self._cudf_data[:self.half]['low'],
                                       self._cudf_data[:self.half]['close'],
                                       self._cudf_data[:self.half]['volume'],
                                       10, 20)
        computed = o[:self.half]['CH_OS_10_20'].to_array('pandas')
        ref = r_cudf.to_array('pandas')
        err = np.abs(computed[~np.isnan(computed)] - ref[~np.isnan(ref)]).max()
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        r_cudf = gi.chaikin_oscillator(self._cudf_data[self.half:]['high'],
                                       self._cudf_data[self.half:]['low'],
                                       self._cudf_data[self.half:]['close'],
                                       self._cudf_data[self.half:]['volume'],
                                       10, 20)
        computed = o[self.half:]['CH_OS_10_20'].to_array('pandas')
        ref = r_cudf.to_array('pandas')
        err = np.abs(computed[~np.isnan(computed)] - ref[~np.isnan(ref)]).max()
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        # check bollinger bands computation
        r_cudf = gi.bollinger_bands(self._cudf_data[:self.half]['close'], 10)
        computed = o[:self.half]["BO_BA_b1_10"].to_array('pandas')
        ref = r_cudf.b1.to_array('pandas')
        err = np.abs(computed[~np.isnan(computed)] - ref[~np.isnan(ref)]).max()
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        computed = o[:self.half]["BO_BA_b2_10"].to_array('pandas')
        ref = r_cudf.b2.to_array('pandas')
        err = np.abs(computed[~np.isnan(computed)] - ref[~np.isnan(ref)]).max()
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        r_cudf = gi.bollinger_bands(self._cudf_data[self.half:]['close'], 10)
        computed = o[self.half:]["BO_BA_b1_10"].to_array('pandas')
        ref = r_cudf.b1.to_array('pandas')
        err = np.abs(computed[~np.isnan(computed)] - ref[~np.isnan(ref)]).max()
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)

        computed = o[self.half:]["BO_BA_b2_10"].to_array('pandas')
        ref = r_cudf.b2.to_array('pandas')
        err = np.abs(computed[~np.isnan(computed)] - ref[~np.isnan(ref)]).max()
        msg = "bad error %f\n" % (err,)
        self.assertTrue(np.isclose(err, 0, atol=1e-6), msg)


if __name__ == '__main__':
    unittest.main()
