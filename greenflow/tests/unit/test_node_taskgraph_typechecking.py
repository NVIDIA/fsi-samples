'''
greenflow Node in TaskGraph Columns Validation Unit Tests

To run unittests:

# Using standard library unittest

python -m unittest -v
python -m unittest tests/unit/test_node_taskgraph_typechecking.py -v
python -m unittest discover -v tests

or

# Using pytest
# "conda install pytest" or "pip install pytest"
pytest -v tests
pytest -v tests/unit/test_node_taskgraph_typechecking.py

'''
import unittest
import copy
import warnings

from greenflow.dataframe_flow import (
    Node, PortsSpecSchema, NodePorts, MetaData)
from greenflow.dataframe_flow import (TaskSpecSchema, TaskGraph)

from .utils import make_orderer

ordered, compare = make_orderer()
unittest.defaultTestLoader.sortTestMethodsUsing = compare


class MyList(list):
    pass


class NodeNumGen(Node):

    def ports_setup(self):
        ptype = self.conf.get('port_type', list)
        output_ports = {'numlist': {PortsSpecSchema.port_type: ptype}}
        return NodePorts(outports=output_ports)

    def meta_setup(self):
        colsopt = self.conf['columns_option']

        cols = {
            'listnums': {'list': 'numbers'},
            'mylistnums': {'list': 'numbers'},
            'rangenums': {'range': 'numbers'},
            'listnotnums': {'list': 'notnumbers'},
        }.get(colsopt)
        return MetaData(inports={}, outports={'numlist': cols})

    def process(self, inputs):
        colsopt = self.conf['columns_option']
        outopt = self.conf.get('out_type', colsopt)
        rng = range(10)

        # get callables according to desired type
        out = {
            'listnums': lambda: list(rng),
            'mylistnums': lambda: MyList(rng),
            'rangenums': lambda: rng,
            'listnotnums': lambda: [str(ii) for ii in rng],
        }.get(outopt)

        return {'numlist': out()}


class NodeNumProc(Node):
    def ports_setup(self):
        ptype = self.conf.get('port_type', list)
        inports = {'inlist': {PortsSpecSchema.port_type: ptype}}
        outports = {'sum': {PortsSpecSchema.port_type: float}}
        return NodePorts(inports=inports, outports=outports)

    def meta_setup(self):
        required = {'inlist': {'list': 'numbers'}}
        columns_out = {'sum': {'element': 'number'}}
        return MetaData(inports=required, outports=columns_out)

    def process(self, inputs):
        inlist = inputs['inlist']
        return {'sum': float(sum(inlist))}


class TestNodeTaskGraphTypechecking(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=DeprecationWarning)

        self.numgen_spec = {
            TaskSpecSchema.task_id: 'numgen',
            TaskSpecSchema.node_type: NodeNumGen,
            TaskSpecSchema.conf: {},
            TaskSpecSchema.inputs: {}
        }

        self.numproc_spec = {
            TaskSpecSchema.task_id: 'numproc',
            TaskSpecSchema.node_type: NodeNumProc,
            TaskSpecSchema.conf: {},
            TaskSpecSchema.inputs: {
                'inlist': 'numgen.numlist'
            }
        }

    def tearDown(self):
        pass

    @ordered
    def test_columns_name_mismatch(self):
        numgen_spec = copy.deepcopy(self.numgen_spec)
        numproc_spec = copy.deepcopy(self.numproc_spec)

        numgen_spec[TaskSpecSchema.conf] = {'columns_option': 'rangenums'}

        tspec_list = [numgen_spec, numproc_spec]
        tgraph_invalid = TaskGraph(tspec_list)

        with self.assertRaises(LookupError) as cm:
            tgraph_invalid.run(['numproc.sum'])
        outerr_msg = '{}'.format(cm.exception)

        errmsg = 'Task "numproc" missing required column "list" from '\
            '"numgen.numlist".'
        self.assertIn(errmsg, outerr_msg)

    @ordered
    def test_columns_type_mismatch(self):
        numgen_spec = copy.deepcopy(self.numgen_spec)
        numproc_spec = copy.deepcopy(self.numproc_spec)

        numgen_spec[TaskSpecSchema.conf] = {'columns_option': 'listnotnums'}

        tspec_list = [numgen_spec, numproc_spec]
        tgraph_invalid = TaskGraph(tspec_list)

        with self.assertRaises(LookupError) as cm:
            tgraph_invalid.run(['numproc.sum'])
        outerr_msg = '{}'.format(cm.exception)

        errmsg = 'Task "numproc" column "list" expected type "numbers" got '\
            'type "notnumbers" instead.'
        self.assertIn(errmsg, outerr_msg)

    @ordered
    def test_ports_output_type_mismatch(self):
        numgen_spec = copy.deepcopy(self.numgen_spec)
        numproc_spec = copy.deepcopy(self.numproc_spec)

        numgen_spec[TaskSpecSchema.conf] = {
            'columns_option': 'listnums',
            'out_type': 'rangenums'
        }

        tspec_list = [numgen_spec, numproc_spec]
        tgraph_invalid = TaskGraph(tspec_list)

        with self.assertRaises(TypeError) as cm:
            tgraph_invalid.run(['numproc.sum'])
        outerr_msg = '{}'.format(cm.exception)

        errmsg = 'Node "numgen" output port "numlist" produced wrong type '\
            '"<class \'range\'>". Expected type "[<class \'list\'>]"'
        self.assertEqual(errmsg, outerr_msg)

    @ordered
    def test_ports_connection_type_mismatch(self):
        numgen_spec = copy.deepcopy(self.numgen_spec)
        numproc_spec = copy.deepcopy(self.numproc_spec)

        numgen_spec[TaskSpecSchema.conf] = {'columns_option': 'listnums'}

        numproc_spec[TaskSpecSchema.conf] = {'port_type': range}

        tspec_list = [numgen_spec, numproc_spec]
        tgraph_invalid = TaskGraph(tspec_list)

        with self.assertRaises(TypeError) as cm:
            tgraph_invalid.run(['numproc.sum'])
        outerr_msg = '{}'.format(cm.exception)

        errmsg = 'Connected nodes do not have matching port types. '\
            'Fix port types.'
        self.assertIn(errmsg, outerr_msg)

    @ordered
    def test_ports_connection_subclass_type_mismatch(self):
        numgen_spec = copy.deepcopy(self.numgen_spec)
        numproc_spec = copy.deepcopy(self.numproc_spec)

        numgen_spec[TaskSpecSchema.conf] = {'columns_option': 'listnums'}
        numproc_spec[TaskSpecSchema.conf] = {'port_type': MyList}

        tspec_list = [numgen_spec, numproc_spec]
        tgraph_invalid = TaskGraph(tspec_list)

        with self.assertRaises(TypeError) as cm:
            tgraph_invalid.run(['numproc.sum'])
        outerr_msg = '{}'.format(cm.exception)

        errmsg = 'Connected nodes do not have matching port types. '\
            'Fix port types.'
        self.assertIn(errmsg, outerr_msg)

    @ordered
    def test_ports_connection_subclass_type_match(self):
        numgen_spec = copy.deepcopy(self.numgen_spec)
        numproc_spec = copy.deepcopy(self.numproc_spec)

        numgen_spec[TaskSpecSchema.conf] = {
            'port_type': MyList,
            'columns_option': 'mylistnums'
        }
        numproc_spec[TaskSpecSchema.conf] = {'port_type': list}

        tspec_list = [numgen_spec, numproc_spec]
        tgraph_valid = TaskGraph(tspec_list)

        sumout, = tgraph_valid.run(['numproc.sum'])

        self.assertEqual(sumout, 45)

    @ordered
    def test_columns_and_ports_types_match(self):
        numgen_spec = copy.deepcopy(self.numgen_spec)
        numproc_spec = copy.deepcopy(self.numproc_spec)

        numgen_spec[TaskSpecSchema.conf] = {'columns_option': 'listnums'}

        tspec_list = [numgen_spec, numproc_spec]
        tgraph_valid = TaskGraph(tspec_list)

        sumout, = tgraph_valid.run(['numproc.sum'])

        self.assertEqual(sumout, 45)


if __name__ == '__main__':
    unittest.main()
