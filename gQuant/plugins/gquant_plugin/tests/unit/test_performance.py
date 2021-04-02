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

    def setUp(self):
        warnings.filterwarnings('ignore', message='numpy.ufunc size changed')
        os.environ['MODULEPATH'] = 'modules'
        self.ports_setup_ref = {
            'ports_setup.datetimeFilterNode.py': 2,
            'ports_setup.minNode.py': 1,
            'ports_setup.maxNode.py': 1,
            'ports_setup.valueFilterNode.py': 1,
            'ports_setup.renameNode.py': 6,
            'ports_setup.assetIndicatorNode.py': 1,
            'ports_setup.dropNode.py': 3,
            'ports_setup.indicatorNode.py': 1,
            'ports_setup.normalizationNode.py': 2,
            'ports_setup.addSignIndicator.py': 3,
            'ports_setup.onehotEncoding.py': 1,
            'ports_setup.persistNode.py': 2,
            'ports_setup.xgboostStrategyNode.py': 2,
            'ports_setup.averageNode.py': 1,
            'ports_setup.leftMergeNode.py': 6,
            'ports_setup.returnFeatureNode.py': 1,
            'ports_setup.sortNode.py': 2,
            'ports_setup.simpleAveragePortOpt.py': 2,
            'ports_setup.compositeNode.py': 4,
            'ports_setup.splitDataNode.py': 2,
            'ports_setup.xgboostNode.py': 8,
            'ports_setup.classificationGenerator.py': 2,
            'ports_setup.simpleBackTest.py': 2,
            'ports_setup.csvStockLoader.py': 3,
            'ports_setup.importanceCurve.py': 1,
            'ports_setup.rocCurveNode.py': 2,
            'ports_setup.sharpeRatioNode.py': 2,
            'ports_setup.cumReturnNode.py': 2,
            'ports_setup.taskGraph.py': 5,
            'ports_setup._node_flow.py': 123,
            'ports_setup.simpleNodeMixin.py': 61
        }
        self.meta_setup_ref = {
            'meta_setup.datetimeFilterNode.py': 2,
            'meta_setup.minNode.py': 1,
            'meta_setup.maxNode.py': 1,
            'meta_setup.valueFilterNode.py': 1,
            'meta_setup.renameNode.py': 3,
            'meta_setup.assetIndicatorNode.py': 1,
            'meta_setup.dropNode.py': 3,
            'meta_setup.indicatorNode.py': 1,
            'meta_setup.normalizationNode.py': 2,
            'meta_setup.addSignIndicator.py': 3,
            'meta_setup.onehotEncoding.py': 1,
            'meta_setup.persistNode.py': 2,
            'meta_setup.xgboostStrategyNode.py': 2,
            'meta_setup.averageNode.py': 1,
            'meta_setup.leftMergeNode.py': 3,
            'meta_setup.returnFeatureNode.py': 1,
            'meta_setup.sortNode.py': 2,
            'meta_setup.simpleAveragePortOpt.py': 2,
            'meta_setup.compositeNode.py': 4,
            'meta_setup.splitDataNode.py': 2,
            'meta_setup.xgboostNode.py': 4,
            'meta_setup.classificationGenerator.py': 2,
            'meta_setup.simpleBackTest.py': 2,
            'meta_setup.csvStockLoader.py': 3,
            'meta_setup.importanceCurve.py': 1,
            'meta_setup.rocCurveNode.py': 2,
            'meta_setup.sharpeRatioNode.py': 2,
            'meta_setup.cumReturnNode.py': 2,
            'meta_setup.taskGraph.py': 5,
            'meta_setup.node.py': 5,
            'meta_setup._node_flow.py': 63,
            'meta_setup.simpleNodeMixin.py': 47
        }

    def tearDown(self):
        pass

    @ordered
    def test_performance(self):
        '''Test frac diff method'''
        profiler = cProfile.Profile()
        profiler.enable()
        graph = TaskGraph.load_taskgraph(
            'taskgraphs/xgboost_example/xgboost_stock.gq.yaml')
        graph.build()
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('ncalls')
        keys = [k for k in stats.stats.keys() if k[-1] == 'ports_setup']
        for key in keys:
            dict_key = key[-1]+'.'+key[0].split('/')[-1]
            print("{}.{}\tis called {}({}) times.".format(
                key[0].split('/')[-1].split('.')[0], key[-1],
                stats.stats[key][0], self.ports_setup_ref[dict_key]))
            self.assertTrue(stats.stats[key][0],
                            self.ports_setup_ref[dict_key])
        keys = [k for k in stats.stats.keys() if k[-1] == 'meta_setup']
        print()
        for key in keys:
            dict_key = key[-1]+'.'+key[0].split('/')[-1]
            print("{}.{}\tis called {}({}) times.".format(
                key[0].split('/')[-1].split('.')[0], key[-1],
                stats.stats[key][0], self.meta_setup_ref[dict_key]))
            self.assertTrue(stats.stats[key][0], self.meta_setup_ref[dict_key])


if __name__ == '__main__':
    unittest.main()
