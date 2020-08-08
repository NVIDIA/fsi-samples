'''
gQuant Node API Unit Tests

To run unittests:

# Using standard library unittest

python -m unittest -v
python -m unittest tests/unit/test_node_api.py -v

or

python -m unittest discover <test_directory>
python -m unittest discover -s <directory> -p 'test_*.py'

# Using pytest
# "conda install pytest" or "pip install pytest"
pytest -v tests
pytest -v tests/unit/test_node_api.py

'''
import os
import unittest

from gquant.dataframe_flow import TaskSpecSchema
from gquant.dataframe_flow.task import Task
from gquant.dataframe_flow._node import _Node
from gquant.dataframe_flow.node import (Node, _PortsMixin)
from gquant.dataframe_flow._node_flow import NodeTaskGraphMixin

from .utils import make_orderer

ordered, compare = make_orderer()
unittest.defaultTestLoader.sortTestMethodsUsing = compare


class TestNodeAPI(unittest.TestCase):

    def setUp(self):
        custom_module = '{}/custom_port_nodes.py'.format(
            os.path.dirname(os.path.realpath(__file__)))

        points_task_spec = {
            TaskSpecSchema.task_id: 'points_task',
            TaskSpecSchema.node_type: 'PointNode',
            TaskSpecSchema.filepath: custom_module,
            TaskSpecSchema.conf: {'npts': 1000},
            TaskSpecSchema.inputs: {}
        }

        self.points_task = Task(points_task_spec)

        distance_task_spec = {
            TaskSpecSchema.task_id: 'distance_by_cudf',
            TaskSpecSchema.node_type: 'DistanceNode',
            TaskSpecSchema.filepath: custom_module,
            TaskSpecSchema.conf: {},
            TaskSpecSchema.inputs: {
                'points_df_in': 'points_task.points_df_out'
            }
        }

        self.distance_task = Task(distance_task_spec)

        points_noports_task_spec = {
            TaskSpecSchema.task_id: 'points_noport_task',
            TaskSpecSchema.node_type: 'PointNoPortsNode',
            TaskSpecSchema.filepath: custom_module,
            TaskSpecSchema.conf: {'npts': 1000},
            TaskSpecSchema.inputs: {}
        }

        self.points_noports_task = Task(points_noports_task_spec)

    def tearDown(self):
        pass

    @ordered
    def test_node_instantiation(self):
        '''Test node instantiation.

            1. Test that you cannot instantiate an abstract base class without
            first implementing the methods requiring override.

            2. Check for the base and base mixin classes in a Node class
            implementation.
        '''
        points_task = self.points_task

        # assert cannot instantiate Node without overriding columns_setup
        # and process
        with self.assertRaises(TypeError) as cm:
            _ = Node(points_task)
        err_msg = '{}'.format(cm.exception)
        self.assertEqual(
            err_msg,
            "Can't instantiate abstract class Node with abstract methods "
            "columns_setup, process")

        points_node = points_task.get_node_obj()

        self.assertIsInstance(points_node, _Node)
        self.assertIsInstance(points_node, Node)
        self.assertIsInstance(points_node, _PortsMixin)
        self.assertNotIsInstance(points_node, NodeTaskGraphMixin)

        points_node = points_task.get_node_obj(tgraph_mixin=True)
        self.assertIsInstance(points_node, NodeTaskGraphMixin)

    @ordered
    def test_node_ports(self):
        '''Test the ports related APIs such as existence of ports, input ports,
        and output ports.
        '''

        distance_node = self.distance_task.get_node_obj()
        iports = distance_node._get_input_ports()
        oports = distance_node._get_output_ports()

        self.assertEqual(iports, ['points_df_in'])
        self.assertEqual(oports, ['distance_df', 'distance_abs_df'])


if __name__ == '__main__':
    unittest.main()
