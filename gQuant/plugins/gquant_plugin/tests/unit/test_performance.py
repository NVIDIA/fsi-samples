'''
Performance Unit Tests

To run unittests:

# Using standard library unittest

python -m unittest -v
python -m unittest tests/unit/test_performance.py -v

or

python -m unittest discover <test_directory>
python -m unittest discover -s <directory> -p 'test_*.py'

# Using pytest
# "conda install pytest" or "pip install pytest"
pytest -v tests
pytest -v tests/unit/test_performance.py

'''

import unittest
from greenflow import TaskGraph
from .utils import make_orderer
import cProfile
import pstats
import warnings
import os

ordered, compare = make_orderer()
unittest.defaultTestLoader.sortTestMethodsUsing = compare


class TestPerformance(unittest.TestCase):
    '''Profile calls to ports_setup and meta_setup.'''

    def setUp(self):
        warnings.filterwarnings('ignore', message='numpy.ufunc size changed')
        dirnamefn = os.path.dirname
        topdir = dirnamefn(dirnamefn(dirnamefn(os.path.realpath(__file__))))
        os.environ['MODULEPATH'] = str(topdir) + '/modules'
        os.environ['GREENFLOW_CONFIG'] = str(topdir) + '/greenflowrc'

        self.ports_setup_ref = {
            'ports_setup.compositeNode.py': 4,
            'ports_setup.classificationGenerator.py': 2,
            'ports_setup.csvStockLoader.py': 3,
            'ports_setup.taskGraph.py': 5,
            'ports_setup._node_flow.py': 320,
            'ports_setup.template_node_mixin.py': 77,
            'ports_setup_ext._node_taskgraph_extension_mixin.py': 77
        }

        self.meta_setup_ref = {
            'meta_setup.normalizationNode.py': 2,
            'meta_setup.compositeNode.py': 4,
            'meta_setup.classificationGenerator.py': 2,
            'meta_setup.simpleBackTest.py': 2,
            'meta_setup.csvStockLoader.py': 3,
            'meta_setup.taskGraph.py': 5,
            'meta_setup.node.py': 5,
            'meta_setup._node_flow.py': 177,
            'meta_setup.template_node_mixin.py': 47,
            'meta_setup_ext._node_taskgraph_extension_mixin.py': 47,
        }

        tgraphpath = str(topdir) + \
            '/taskgraphs/xgboost_example/xgboost_stock.gq.yaml'
        profiler = cProfile.Profile()
        profiler.enable()
        graph = TaskGraph.load_taskgraph(tgraphpath)
        graph.build()
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('ncalls')

        self.stats = stats

    def tearDown(self):
        pass

    @ordered
    def test_ports_setup_performance(self):
        stats = self.stats
        statkeys = self.stats.stats.keys()
        keys = [k for k in statkeys if k[-1] in ('ports_setup',)] + \
            [k for k in statkeys if k[-1] in ('ports_setup_ext',)]
        for key in keys:
            dict_key = key[-1]+'.'+key[0].split('/')[-1]
            msg = "{}.{} is called {} (expected {}) times.".format(
                key[0].split('/')[-1].split('.')[0], key[-1],
                stats.stats[key][0], self.ports_setup_ref[dict_key])
            self.assertTrue(
                stats.stats[key][0] == self.ports_setup_ref[dict_key], msg)

    @ordered
    def test_meta_setup_performance(self):
        stats = self.stats
        statkeys = self.stats.stats.keys()

        keys = [k for k in statkeys if k[-1] in ('meta_setup',)] + \
            [k for k in statkeys if k[-1] in ('meta_setup_ext',)]
        for key in keys:
            dict_key = key[-1] + '.' + key[0].split('/')[-1]
            msg = "{}.{} is called {} (expected {}) times.".format(
                key[0].split('/')[-1].split('.')[0], key[-1],
                stats.stats[key][0], self.meta_setup_ref[dict_key])
            self.assertTrue(
                stats.stats[key][0] == self.meta_setup_ref[dict_key], msg)


if __name__ == '__main__':
    unittest.main()
