import ast
from types import ModuleType

import numpy as np
import cupy as cp

from greenflow.dataframe_flow import (Node, PortsSpecSchema, ConfSchema)
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin

__all__ = ['CustomFilterNode']


def compile_user_module(code):
    '''
    Usage:
        # code is some text/string of code to be compiled dynamically.
        code = '\ndef somefn(in1, in2):\n    return in1 + in2\n'
        module_ = compile_user_module(code)
        module_.somefn(5, 6)  # returns 11 per def of somefn
    '''
    # https://stackoverflow.com/questions/19850143/how-to-compile-a-string-of-python-code-into-a-module-whose-functions-can-be-call
    # https://stackoverflow.com/questions/39379331/python-exec-a-code-block-and-eval-the-last-line
    block = ast.parse(code, mode='exec')

    module_ = ModuleType('user_module')
    exec(compile(block, '<string>', mode='exec'), module_.__dict__)

    return module_


class CustomFilterNode(TemplateNodeMixin, Node):
    def init(self):
        TemplateNodeMixin.init(self)

        port_type = PortsSpecSchema.port_type
        inports = {'signal': {port_type: [cp.ndarray, np.ndarray]}}
        outports = {'signal_out': {port_type: '${port:signal}'}}
        self.template_ports_setup(in_ports=inports, out_ports=outports)

        meta_outports = {'signal_out': {}}
        self.template_meta_setup(out_ports=meta_outports)

    def conf_schema(self):
        json = {
            'title': 'Custom Filter Node.',
            'type': 'object',
            'description': 'Custom filter logic. CAUTION: Only run trusted '
                'code.',  # noqa: E131,E501
            'properties': {
                'pycode': {
                    'type': 'string',
                    'title': 'Signal Code - pycode',
                    'description': 'Enter python code to filter a signal. '
                        'The code must have a function with the following '  # noqa: E131,E501
                        'name and signature: def custom_filter(signal, conf). '
                        'The ``signal`` is a cp or np array. The ``conf`` '
                        'is the node\'s configuration dictionary. Besides '
                        '"pycode" custom conf fields are not not exposed via '
                        'UI. If anything needs to be set do it '
                        'programmatically via TaskSpecSchema. The '
                        '`custom_filter` function must return a processed '
                        'signal of same type as input signal.'
                },
            },
            # 'required': ['pycode'],
        }
        ui = {'pycode': {'ui:widget': 'textarea'}}
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        pycode = self.conf.get('pycode')

        if not pycode:
            raise RuntimeError('Task id: {}; Node type: {}\n'
                               'No code provided. Nothing to output.'
                               .format(self.uid, 'CustomFilterNode'))

        signal = inputs['signal']
        module_ = compile_user_module(pycode)
        if not hasattr(module_, 'custom_filter'):
            raise RuntimeError(
                'Task id: {}; Node type: {}\n'
                'Pycode does not define "custom_filter" function.\n'
                'Pycode provided:\n{}'
                .format(self.uid, 'CustomFilterNode', pycode))

        out = module_.custom_filter(signal, self.conf)
        return {'signal_out': out}
