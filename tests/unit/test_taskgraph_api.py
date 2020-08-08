'''
gQuant TaskGraph API Unit Tests

To run unittests:

# Using standard library unittest

python -m unittest -v
python -m unittest tests/unit/test_taskgraph_api.py -v

or

python -m unittest discover <test_directory>
python -m unittest discover -s <directory> -p 'test_*.py'

# Using pytest
# "conda install pytest" or "pip install pytest"
pytest -v tests
pytest -v tests/unit/test_taskgraph_api.py

'''
import os
import shutil
import tempfile
from difflib import context_diff
import yaml
from io import StringIO
import warnings
import unittest

from gquant.dataframe_flow import (TaskSpecSchema, TaskGraph)
from gquant.dataframe_flow.task import DEFAULT_MODULE  # noqa: F401
from gquant.dataframe_flow import Node

from .utils import make_orderer

ordered, compare = make_orderer()
unittest.defaultTestLoader.sortTestMethodsUsing = compare


TASKGRAPH_YAML = \
    '''- id: points_task
  type: PointNode
  conf:
    npts: 1000
  inputs: []
- id: distance_by_cudf
  type: DistanceNode
  conf: {}
  inputs:
    points_df_in: points_task.points_df_out
'''


class TestTaskGraphAPI(unittest.TestCase):
    def setUp(self):
        import gc  # python garbage collector
        import cudf

        # warmup
        s = cudf.Series([1, 2, 3, None, 4], nan_as_null=False)
        del(s)
        gc.collect()

        os.environ['GQUANT_PLUGIN_MODULE'] = 'tests.unit.custom_port_nodes'

        points_task_spec = {
            TaskSpecSchema.task_id: 'points_task',
            TaskSpecSchema.node_type: 'PointNode',
            TaskSpecSchema.conf: {'npts': 1000},
            TaskSpecSchema.inputs: []
        }

        distance_task_spec = {
            TaskSpecSchema.task_id: 'distance_by_cudf',
            TaskSpecSchema.node_type: 'DistanceNode',
            TaskSpecSchema.conf: {},
            TaskSpecSchema.inputs: {
                'points_df_in': 'points_task.points_df_out'
            }
        }

        tspec_list = [points_task_spec, distance_task_spec]

        self.tgraph = TaskGraph(tspec_list)

        # Create a temporary directory
        self._test_dir = tempfile.mkdtemp()
        os.environ['GQUANT_CACHE_DIR'] = os.path.join(self._test_dir, '.cache')

    def tearDown(self):
        global DEFAULT_MODULE
        os.environ['GQUANT_PLUGIN_MODULE'] = DEFAULT_MODULE
        os.environ['GQUANT_CACHE_DIR'] = Node.cache_dir
        shutil.rmtree(self._test_dir)

    @ordered
    def test_viz_graph(self):
        '''Test taskgraph to networkx graph conversion for graph visualization.
        '''
        nx_graph = self.tgraph.viz_graph(show_ports=True)
        nx_nodes = [
            'points_task', 'points_task.points_df_out',
            'distance_by_cudf', 'distance_by_cudf.distance_df'
        ]
        nx_nodes = ['points_task', 'points_task.points_df_out',
                    'points_task.points_ddf_out',
                    'distance_by_cudf', 'distance_by_cudf.distance_df',
                    'distance_by_cudf.distance_abs_df']
        nx_edges = [('points_task', 'points_task.points_df_out'),
                    ('points_task', 'points_task.points_ddf_out'),
                    ('points_task.points_df_out', 'distance_by_cudf'),
                    ('distance_by_cudf', 'distance_by_cudf.distance_df'),
                    ('distance_by_cudf', 'distance_by_cudf.distance_abs_df')]
        self.assertEqual(list(nx_graph.nodes), nx_nodes)
        self.assertEqual(list(nx_graph.edges), nx_edges)

    @ordered
    def test_build(self):
        '''Test build of a taskgraph and that all inputs and outputs are set
        for the tasks withink a taskgraph.
        '''
        self.tgraph.build()

        points_node = self.tgraph['points_task']
        distance_node = self.tgraph['distance_by_cudf']

        onode_info = {
            'to_node': distance_node,
            'to_port': 'points_df_in',
            'from_port': 'points_df_out'
        }
        self.assertIn(onode_info, points_node.outputs)

        onode_cols = {'points_df_out': {'x': 'float64', 'y': 'float64'},
                      'points_ddf_out': {'x': 'float64', 'y': 'float64'}}
        self.assertEqual(onode_cols, points_node.columns_setup())

        inode_info = {
            'from_node': points_node,
            'from_port': 'points_df_out',
            'to_port': 'points_df_in'
        }
        self.assertIn(inode_info, distance_node.inputs)

        inode_in_cols = {
            'points_df_in': {
                'x': 'float64',
                'y': 'float64'
            }
        }
        self.assertEqual(inode_in_cols, distance_node.get_input_columns())

        inode_out_cols = {'distance_df': {'distance_cudf': 'float64',
                                          'x': 'float64',
                                          'y': 'float64'},
                          'distance_abs_df': {'distance_abs_cudf': 'float64',
                                              'x': 'float64', 'y': 'float64'}}
        self.assertEqual(inode_out_cols, distance_node.columns_setup())

    @ordered
    def test_run(self):
        '''Test that a taskgraph can run successfully.
        '''
        outlist = ['distance_by_cudf.distance_df']
        # Using numpy random seed to get repeatable and deterministic results.
        # For seed 2335 should get something around 761.062831178.
        replace_spec = {
            'points_task': {
                TaskSpecSchema.conf: {
                    'npts': 1000,
                    'nseed': 2335
                }
            }
        }
        (dist_df_w_cudf, ) = self.tgraph.run(
            outputs=outlist, replace=replace_spec)
        dist_sum = dist_df_w_cudf['distance_cudf'].sum()
        # self.assertAlmostEqual(dist_sum, 0.0, places, msg, delta)
        self.assertAlmostEqual(dist_sum, 761.062831178)  # match to 7 places

    @ordered
    def test_save(self):
        '''Test that a taskgraph can be save to a yaml file.
        '''
        workflow_file = os.path.join(self._test_dir,
                                     'test_save_taskgraph.yaml')
        self.tgraph.save_taskgraph(workflow_file)

        with open(workflow_file) as wf:
            workflow_str = wf.read()

        # verify the workflow contentst same as expected. Empty list if same.
        global TASKGRAPH_YAML
        cdiff = list(context_diff(TASKGRAPH_YAML, workflow_str))
        cdiff_empty = cdiff == []

        err_msg = 'Taskgraph yaml contents do not match expected results.\n'\
            'SHOULD HAVE SAVED:\n\n'\
            '{wyaml}\n\n'\
            'INSTEAD FILE CONTAINS:\n\n'\
            '{fcont}\n\n'\
            'DIFF:\n\n'\
            '{diff}'.format(wyaml=TASKGRAPH_YAML, fcont=workflow_str,
                            diff=''.join(cdiff))

        self.assertTrue(cdiff_empty, err_msg)

    @ordered
    def test_load(self):
        '''Test that a taskgraph can be loaded from a yaml file.
        '''
        workflow_file = os.path.join(self._test_dir,
                                     'test_load_taskgraph.yaml')

        global TASKGRAPH_YAML
        with open(workflow_file, 'w') as wf:
            wf.write(TASKGRAPH_YAML)

        tspec_list = [task._task_spec for task in self.tgraph]

        tgraph = TaskGraph.load_taskgraph(workflow_file)
        all_tasks_exist = True
        for task in tgraph:
            if task._task_spec not in tspec_list:
                all_tasks_exist = False
                break

        with StringIO() as yf:
            yaml.dump(tspec_list, yf,
                      default_flow_style=False, sort_keys=False)
            yf.seek(0)

            err_msg = 'Load taskgraph failed. Missing expected task items.\n'\
                'EXPECTED TASKGRAPH YAML:\n\n'\
                '{wyaml}\n\n'\
                'GOT TASKS FORMATTED AS YAML:\n\n'\
                '{tlist}\n\n'.format(wyaml=TASKGRAPH_YAML, tlist=yf.read())

            self.assertTrue(all_tasks_exist, err_msg)

    @ordered
    def test_save_load_cache(self):
        '''Test caching of tasks outputs within a taskgraph.

            1. Save points_task output to cache when running the taskgraph.
            2. Load points_task df from cache when running the taskgraph.
        '''
        replace_spec = {'points_task': {TaskSpecSchema.save: True}}
        outlist = ['distance_by_cudf.distance_df']

        with warnings.catch_warnings():
            # ignore UserWarning: Using CPU via Pandas to write HDF dataset
            warnings.filterwarnings(
                'ignore',
                message='Using CPU via Pandas to write HDF dataset',
                category=UserWarning,)
            # ignore RuntimeWarning: numpy.ufunc size changed
            warnings.filterwarnings('ignore',
                                    category=RuntimeWarning,
                                    message='numpy.ufunc size changed')
            (_, ) = self.tgraph.run(outputs=outlist, replace=replace_spec)

        cache_dir = os.path.join(self._test_dir, '.cache', 'points_task.hdf5')
        self.assertTrue(os.path.exists(cache_dir))

        replace_spec = {'points_task': {TaskSpecSchema.load: True}}
        with warnings.catch_warnings():
            # ignore UserWarning: Using CPU via Pandas to read HDF dataset
            warnings.filterwarnings(
                'ignore',
                message='Using CPU via Pandas to read HDF dataset',
                category=UserWarning)
            (_, ) = self.tgraph.run(outputs=outlist, replace=replace_spec)


if __name__ == '__main__':
    unittest.main()
