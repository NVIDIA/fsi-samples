import numpy as np
import cupy as cp

from cusignal.spectral_analysis import welch as cuwelch
from scipy.signal.spectral import welch as siwelch

from greenflow.dataframe_flow import (
    Node, NodePorts, PortsSpecSchema, ConfSchema, MetaData)

from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin

from ..windows import _WINS_CONFIG

__all__ = ['WelchPSD_Node']

_WELCH_DESC = '''Estimate power spectral density using Welch's method. Welch's
method computes an estimate of the power spectral density by dividing the data
into overlapping segments, computing a modified periodogram for each segment
and averaging the periodograms.
Returns -  freqs:ndarray Array of frequencies;
Pxx:ndarray Power spectral density or power spectrum of signal.
'''


class WelchPSD_Node(TemplateNodeMixin, Node):
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
            'psd': {PortsSpecSchema.port_type: '${port:signal}'},
            'freqs': {PortsSpecSchema.port_type: '${port:signal}'},
        }
        self.template_ports_setup(in_ports=inports, out_ports=outports)

        meta_outports = {'psd': {}, 'freqs': {}}
        self.template_meta_setup(out_ports=meta_outports)

    def conf_schema(self):
        windows_enum = list(_WINS_CONFIG.keys())
        detrend_enum = ['constant', 'linear', 'false']
        scaling_enum = ['density', 'spectrum']
        average_enum = ['mean', 'median']
        json = {
            'title': 'Welch Power Spectral Density Node',
            'type': 'object',
            'description': _WELCH_DESC,
            'properties': {
                'samplerate': {
                    'type': 'number',
                    'description': 'fs : float, optional; Sampling frequency '
                        'of the `x` (input signal) time series. Defaults to '  # noqa: E131,E501
                        '1.0. This can also be passed at input port '
                        '`samplerate`. Port takes precedence over conf.',
                    'default': 1.0
                },
                'window':  {
                    'type': 'string',
                    'description': 'Desired window to use. Alternatively '
                        'pass window via port `window`. In that case its '  # noqa: E131,E501
                        'length must be nperseg. Defaults to a Hann window.',
                    'enum': windows_enum,
                    'default': 'hann'
                },
                'nperseg': {
                    'type': 'integer',
                    'description': 'Length of each segment. Defaults to None, '
                        'but if window is str, is set to 256, and if window '  # noqa: E131,E501
                        'is array_like (passed via port `window`), is set to '
                        'the lesser of this setting or length of the window.',
                },
                'noverlap': {
                    'type': 'integer',
                    'description': 'Number of points to overlap between '
                        'segments. If `None`, ``noverlap = nperseg // 2``. '  # noqa: E131,E501
                        'Defaults to `None`.',
                },
                'nfft': {
                    'type': 'integer',
                    'description': 'Length of the FFT used, if a zero padded '
                        'FFT is desired. If `None`, the FFT length is '  # noqa: E131,E501
                        '`nperseg`. Defaults to `None`.',
                },
                'detrend': {
                    'type': 'string',
                    'description': 'Specifies how to detrend each segment. If '
                        '"constant", only the mean of `data` is subtracted. '  # noqa: E131,E501
                        'If "linear", the result of a linear least-squares '
                        'fit to `data` is subtracted from `data`. If '
                        '`detrend` is `False`, no detrending is done. '
                        'Default is "constant".',
                    'enum': detrend_enum,
                    'default': 'constant'
                },
                'return_onesided': {
                    'type': 'boolean',
                    'description': 'return_onesided - If `True`, return a '
                        'one-sided spectrum for real data. If `False` return '  # noqa: E131,E501
                        'a two-sided spectrum. Defaults to `True`, but for '
                        'complex data, a two-sided spectrum is always '
                        'returned.',
                    'default': True
                },
                'scaling': {
                    'type': 'string',
                    'description': 'Selects between computing the power '
                        'spectral density ("density") where `Pxx` has units '  # noqa: E131,E501
                        'of V**2/Hz and computing the power spectrum '
                        '("spectrum") where `Pxx` has units of V**2, if `x` '
                        'is measured in V and `fs` is measured in Hz. '
                        'Defaults to density',
                    'enum': scaling_enum,
                    'default': 'constant'
                },
                'axis': {
                    'type': 'integer',
                    'description': 'Axis along which the periodogram is '
                        'computed; the default is over the last axis (i.e. '  # noqa: E131,E501
                        '``axis=-1``).',
                    'default': -1
                },
                'average': {
                    'type': 'string',
                    'description': '{"mean", "median"}, optional. Method to '
                        'use when averaging periodograms. Defaults to "mean".',  # noqa: E131,E501
                    'enum': average_enum,
                    'default': 'mean'
                },
                'use_cpu': {
                    'type': 'boolean',
                    'description': 'Use CPU for computation via '
                        'scipy::signal.spectral.welch. Default is False and '  # noqa: E131,E501
                        'runs on GPU via cusignal.',
                    'default': False
                },
            },
        }
        return ConfSchema(json=json)

    def process(self, inputs):
        use_cpu = self.conf.get('use_cpu', False)

        signal = inputs['signal']

        samplerate = self.conf.get('samplerate', 1.0)
        samplerate = inputs.get('samplerate', samplerate)

        window = self.conf.get('window', 'hann')
        window = inputs.get('window', window)

        nperseg = self.conf.get('nperseg', None)
        try:
            nperseg = window.shape[0]
        except Exception:
            pass

        noverlap = self.conf.get('noverlap', None)
        nfft = self.conf.get('nfft', None)

        detrend = self.conf.get('detrend', 'constant')
        if isinstance(detrend, str):
            detrend = False if detrend.lower() in ('false',) else detrend

        return_onesided = self.conf.get('return_onesided', True)
        scaling = self.conf.get('scaling', 'density')
        axis = self.conf.get('axis', -1)
        average = self.conf.get('average', 'mean')

        welch_params = {
            'fs': samplerate,
            'window': window,
            'nperseg': nperseg,
            'noverlap': noverlap,
            'nfft': nfft,
            'detrend': detrend,
            'return_onesided': return_onesided,
            'scaling': scaling,
            'axis': axis,
            'average': average,
        }

        if use_cpu:
            freqs, psd = siwelch(signal, **welch_params)
        else:
            freqs, psd = cuwelch(signal, **welch_params)

        return {'psd': psd, 'freqs': freqs}
