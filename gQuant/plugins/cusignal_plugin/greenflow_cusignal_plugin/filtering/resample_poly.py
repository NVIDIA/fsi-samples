from ast import literal_eval
from fractions import Fraction
import numpy as np
import cupy as cp

from cusignal.filtering.resample import resample_poly as curesamp
from scipy.signal import resample_poly as siresamp

from greenflow.dataframe_flow import (Node, PortsSpecSchema, ConfSchema)
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin

from ..windows import _WINS_CONFIG

__all__ = ['CusignalResamplePolyNode']

_RESAMPLEPOLY_DESC = '''Resample `signal` along the given axis using polyphase
filtering. The signal is upsampled by the factor `up`, a zero-phase low-pass
FIR filter is applied, and then it is downsampled by the factor `down`.
The resulting sample rate is ``up / down`` times the original sample
rate. Values beyond the boundary of the signal are assumed to be zero
during the filtering step. Returns resampled array and new sample rate.
'''


class CusignalResamplePolyNode(TemplateNodeMixin, Node):
    def init(self):
        TemplateNodeMixin.init(self)
        inports = {
            'signal': {PortsSpecSchema.port_type: [cp.ndarray, np.ndarray]},
            'samplerate': {
                PortsSpecSchema.port_type: [int, float, np.float32,
                                            np.float64],
                PortsSpecSchema.optional: True
            },
            'window': {
                PortsSpecSchema.port_type: [cp.ndarray, np.ndarray],
                PortsSpecSchema.optional: True
            },
        }
        outports = {
            'signal_out': {PortsSpecSchema.port_type: '${port:signal}'},
            'samplerate_out': {
                PortsSpecSchema.port_type: [int, float, np.float32,
                                            np.float64],
                PortsSpecSchema.optional: True
            }
        }
        self.template_ports_setup(in_ports=inports, out_ports=outports)

        meta_outports = {'signal_out': {}, 'samplerate_out': {}}
        self.template_meta_setup(out_ports=meta_outports)

    def conf_schema(self):
        padtype_enum = ['constant', 'line', 'mean', 'median', 'maximum',
                        'minimum']
        json = {
            'title': 'Polyphase Filter Resample Node',
            'type': 'object',
            'description': _RESAMPLEPOLY_DESC,
            'properties': {
                'new_samplerate': {
                    'type': 'number',
                    'description': 'Desired sample rate. Specify this or the '
                        'up/down parameters. This is used when `samplerate` '  # noqa: E131,E501
                        'is passed in via ports, otherwise up/down is used. '
                        'If both are set then this takes precedence over '
                        'up/down.'
                },
                'up': {
                    'type': 'integer',
                    'description': 'The upsampling factor.'
                },
                'down': {
                    'type': 'integer',
                    'description': 'The downsampling factor.'
                },
                'axis': {
                    'type': 'integer',
                    'description': 'The axis of `x` that is resampled. '
                        'Default is 0.',  # noqa: E131,E501
                    'default': 0,
                    'minimum': 0,
                },
                'window':  {
                    'type': 'string',
                    'description': 'Desired window to use to design the '
                        'low-pass filter, or the FIR filter coefficients to '  # noqa: E131,E501
                        'employ. Window can be specified as a string, a '
                        'tuple, or a list. If a string choose one of '
                        'available windows. If a tuple refer to '
                        '`cusignal.windows.get_window`. The tuple format '
                        'specifies the first argument as the string name of '
                        'the window, and the next arguments the needed '
                        'parameters. If `window` is a list it is assumed to '
                        'be the FIR filter coefficients. Note that the FIR '
                        'filter is applied after the upsampling step, so it '
                        'should be designed to operate on a signal at a '
                        'sampling frequency higher than the original by a '
                        'factor of `up//gcd(up, down)`. If the port window '
                        'is connected it takes precedence. Default '
                        '("kaiser", 5.0)',
                    'default': '("kaiser", 5.0)'
                },
                'gpupath': {
                    'type': 'boolean',
                    'description': 'gpupath - Optional path for filter design.'
                        ' gpupath == False may be desirable if filter sizes '  # noqa: E131,E501
                        'are small.',
                    'default': True
                },
                'use_cpu': {
                    'type': 'boolean',
                    'description': 'use_cpu - Use CPU for computation via '
                        'scipy::signal.resample_poly. Default is False and '  # noqa: E131,E501
                        'runs on GPU via cusignal.',
                    'default': False
                },
                'padtype': {
                    'type': 'string',
                    'description': 'Only used when `use_cpu` is set. Scipy '
                        'padtype parameter of `resample_poly`. This is not '  # noqa: E131,E501
                        'currently exposed in cusignal.',
                    'enum': padtype_enum,
                    'default': 'constant'
                },
                'cval': {
                    'type': 'number',
                    'description': 'Only used when `use_cpu` is set. Value '
                        'to use if `padtype="constant"`. Default is zero.'  # noqa: E131,E501
                }
            }
        }
        return ConfSchema(json=json)

    def process(self, inputs):
        signal_in = inputs['signal']
        samplerate = inputs.get('samplerate', None)

        new_samplerate = self.conf.get('new_samplerate', None)
        if new_samplerate and samplerate:
            ud = Fraction(new_samplerate / samplerate).limit_denominator()
            up = ud.numerator
            down = ud.denominator
        else:
            up = self.conf['up']
            down = self.conf['down']

        if samplerate:
            samplerate = inputs['samplerate']
            new_samplerate = samplerate * up / down
        else:
            new_samplerate = up / down

        axis = self.conf.get('axis', 0)

        if 'window' in inputs:
            window = input['window']
        else:
            window = self.conf.get('window', ("kaiser", 5.0))
            if isinstance(window, str):
                windows_enum = list(_WINS_CONFIG.keys())
                # window could be a simple string or python code for tuple
                if window not in windows_enum:
                    # window should be a string that is python code
                    # evaluated to a tuple.
                    try:
                        window = literal_eval(window)
                    except Exception:
                        raise RuntimeError('Uknown window: {}'.format(window))

        gpupath = self.conf.get('gpupath', True)

        use_cpu = self.conf.get('use_cpu', False)

        if use_cpu:
            padtype = self.conf.get('padtype', 'constant')
            cval = self.conf.get('cval')
            signal_out = siresamp(
                signal_in, up, down, axis=axis, window=window,
                padtype=padtype, cval=cval)
        else:
            signal_out = curesamp(
                signal_in, up, down, axis=axis, window=window, gpupath=gpupath)

        return {'signal_out': signal_out,
                'samplerate_out': new_samplerate}
