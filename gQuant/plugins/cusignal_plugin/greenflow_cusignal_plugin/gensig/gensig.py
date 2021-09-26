import numpy as np
import cupy as cp
import ast

from greenflow.dataframe_flow import (
    Node, NodePorts, PortsSpecSchema, ConfSchema, MetaData)
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin

__all__ = ['SignalGeneratorNode']


def exec_then_eval(code):
    # https://stackoverflow.com/questions/39379331/python-exec-a-code-block-and-eval-the-last-line
    block = ast.parse(code, mode='exec')

    # assumes last node is an expression
    last = ast.Expression(block.body.pop().value)

    _globals, _locals = {}, {}
    exec(compile(block, '<string>', mode='exec'), _globals, _locals)
    return eval(compile(last, '<string>', mode='eval'), _globals, _locals)


class SignalGeneratorNode(TemplateNodeMixin, Node):
    def init(self):
        TemplateNodeMixin.init(self)
        outports = {
            'out1': {PortsSpecSchema.port_type: [cp.ndarray, np.ndarray]},
            'out2': {
                PortsSpecSchema.port_type: [cp.ndarray, np.ndarray],
                PortsSpecSchema.optional: True
            },
        }
        self.template_ports_setup(out_ports=outports)

        meta_outports = {'out1': {}, 'out2': {}}
        self.template_meta_setup(out_ports=meta_outports)

    def conf_schema(self):
        json = {
            'title': 'Custom Signal Generator Node.',
            'type': 'object',
            'description': 'Inject signals into greenflow taskgraphs. Use '
                'CAUTION. Only run trusted code.',
            'properties': {
                'pycode': {
                    'type': 'string',
                    'title': 'Signal Code',
                    'description': 'Enter python code to generate signal. '
                        'The code must have a dictionary ``myout`` variable '
                        'with keys: out1 and out2. The out2 port is optional. '
                        'The ``myout`` must be the last line. Keep it simple '
                        'please.'
                },
            },
            # 'required': ['pycode'],
        }
        ui = {'pycode': {'ui:widget': 'textarea'}}
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        pycode = self.conf.get('pycode')
        # print('Task id: {}; Node type: {}\nPYCODE:\n{}'.format(
        #     self.uid, 'SignalGeneratorNode', pycode))

        if pycode:
            myout = exec_then_eval(pycode)
            return myout

        raise RuntimeError('Task id: {}; Node type: {}\n'
                           'No pycode provided. Nothing to output.'
                           .format(self.uid, 'SignalGeneratorNode'))
