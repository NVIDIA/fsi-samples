'''
Workflow Serialization Unit Tests

To run unittests:

# Using standard library unittest

python -m unittest -v
python -m unittest tests/unit/test_workflow_serialization.py -v

or

python -m unittest discover <test_directory>
python -m unittest discover -s <directory> -p 'test_*.py'

# Using pytest
# "conda install pytest" or "pip install pytest"
pytest -v tests
pytest -v tests/unit/test_workflow_serialization.py

'''
import os
import warnings
from io import StringIO
import yaml
import shutil
import tempfile
import unittest
from difflib import context_diff
from .utils import make_orderer

ordered, compare = make_orderer()
unittest.defaultTestLoader.sortTestMethodsUsing = compare

# ------------------------------------------- Workflow Serialization Test Cases
WORKFLOW_YAML = \
    '''- id: points
  type: PointNode
  conf: {}
  inputs: []
  filepath: custom_nodes.py
- id: distance
  type: DistanceNode
  conf: {}
  inputs:
  - points
  filepath: custom_nodes.py
- id: node_outputCsv
  type: OutCsvNode
  conf:
    path: symbol_returns.csv
  inputs:
  - distance
'''


class TestWorkflowSerialization(unittest.TestCase):

    def setUp(self):
        # ignore importlib warnings.
        warnings.simplefilter('ignore', category=ImportWarning)
        warnings.simplefilter('ignore', category=DeprecationWarning)

        # some dummy tasks
        task_input = {
            'id': 'points',
            'type': 'PointNode',
            'conf': {},
            'inputs': [],
            'filepath': 'custom_nodes.py'
        }

        task_compute = {
            'id': 'distance',
            'type': 'DistanceNode',
            'conf': {},
            'inputs': ['points'],
            'filepath': 'custom_nodes.py'
        }

        task_output = {
            'id': 'node_outputCsv',
            'type': 'OutCsvNode',
            'conf': {
                'path': 'symbol_returns.csv'
            },
            'inputs': ['distance']
        }
        self._task_list = [task_input, task_compute, task_output]

        # Create a temporary directory
        self._test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self._test_dir)

    @ordered
    def test_save_workflow(self):
        '''Test saving a workflow to yaml:'''
        from gquant.dataframe_flow import TaskGraph
        task_graph = TaskGraph(self._task_list)
        workflow_file = os.path.join(self._test_dir, 'test_save_workflow.yaml')
        task_graph.save(workflow_file)

        with open(workflow_file) as wf:
            workflow_str = wf.read()

        # verify the workflow contentst same as expected. Empty list if same.
        cdiff = list(context_diff(WORKFLOW_YAML, workflow_str))
        cdiff_empty = cdiff == []

        err_msg = 'Workflow yaml contents do not match expected results.\n'\
            'SHOULD HAVE SAVED:\n\n'\
            '{wyaml}\n\n'\
            'INSTEAD FILE CONTAINS:\n\n'\
            '{fcont}\n\n'\
            'DIFF:\n\n'\
            '{diff}'.format(wyaml=WORKFLOW_YAML, fcont=workflow_str,
                            diff=''.join(cdiff))

        self.assertTrue(cdiff_empty, err_msg)

    @ordered
    def test_load_workflow(self):
        '''Test loading a workflow from yaml:'''
        from gquant.dataframe_flow import TaskGraph
        workflow_file = os.path.join(self._test_dir, 'test_save_workflow.yaml')

        with open(workflow_file, 'w') as wf:
            wf.write(WORKFLOW_YAML)

        task_list = TaskGraph.load(workflow_file)
        all_tasks_exist = True
        for t in task_list:
            match = False
            if t._task_spec in self._task_list:
                match = True
            if not match:
                all_tasks_exist = False
                break
        with StringIO() as yf:
            yaml.dump(self._task_list, yf,
                      default_flow_style=False, sort_keys=False)
            yf.seek(0)

            err_msg = 'Load workflow failed. Missing expected task items.\n'\
                'EXPECTED WORKFLOW YAML:\n\n'\
                '{wyaml}\n\n'\
                'GOT TASKS FORMATTED AS YAML:\n\n'\
                '{tlist}\n\n'.format(wyaml=WORKFLOW_YAML, tlist=yf.read())

            self.assertTrue(all_tasks_exist, err_msg)


if __name__ == '__main__':
    # with warnings.catch_warnings():
    #     warnings.simplefilter('ignore', category=ImportWarning)
    #     unittest.main()
    unittest.main()
