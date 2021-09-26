import inspect
import numpy as np
import cupy as cp

import cusignal.windows as cuwin
import scipy.signal.windows as siwin

from greenflow.dataframe_flow import (
    Node, NodePorts, PortsSpecSchema, ConfSchema, MetaData)

from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin

__all__ = ['CusignalWindowNode']


_DEFAULT_WIN_JSON_CONF = {
     'M': {
        'type': 'integer',
        'title': 'M',
        'description': 'Number of points in the output window. If '
            'zero or less, an empty array is returned.'  # noqa: E131,E501
    },
    'sym': {
        'type': 'boolean',
        'title': 'sym',
        'description': 'When True (default), generates a symmetric '
            'window, for use in filter design. When False, generates a '  # noqa: E131,E501
            'periodic window, for use in spectral analysis.',
        'default': True
    }
}

_DEFAULT_WIN_RETDESC = 'Returns - window : ndarray; The window, with the '\
'maximum value normalized to 1 (though the value 1 does not appear if `M` '\
'is even and `sym` is True)'

_WINS_CONFIG = {
    'general_cosine': {
        'json_conf': {
            'a': {
                'type': 'array',
                'items': {'type': 'number'},
                'description': 'Sequence of weighting coefficients. This '
                    'uses the convention of being centered on the origin, '
                    'so these will typically all be positive numbers, not '
                    'alternating sign.'
            },
        },
        'description': 'Generic weighted sum of cosine terms window.',
        'desc-return': ''
    },
    'boxcar': {
        'description': 'Return a boxcar or rectangular window. '
            'Also known as a rectangular window or Dirichlet window, this is '
            'equivalent to no window at all.',
        'desc-return': 'window: ndarray; The window, with the maximum value '
            'normalized to 1.'
    },
    'triang': {
        'description': 'Return a triangular window.'
    },
    'parzen': {
        'description': 'Return a Parzen window.',
        'desc-return': ''
    },
    'bohman': {
        'description': 'Return a Bohman window.'
    },
    'blackman': {
        'description': 'The Blackman window is a taper formed by using the '
            'first three terms of a summation of cosines. It was designed to '
            'have close to the minimal leakage possible.  It is close to '
            'optimal, only slightly worse than a Kaiser window.'
    },
    'nuttall': {
        'description': 'Return a minimum 4-term Blackman-Harris window '
        'according to Nuttall. This variation is also called "Nuttall4c".'
    },
    'blackmanharris': {
        'description': 'Return a minimum 4-term Blackman-Harris window.'
    },
    'flattop': {
        'description': 'Return a flat top window.'
    },
    'bartlett': {
        'description': 'Return a Bartlett window. The Bartlett window is very '
            'similar to a triangular window, except that the end points are '
            'at zero.  It is often used in signal processing for tapering a '
            'signal, without generating too much ripple in the frequency '
            'domain.',
        'desc-return': 'Returns - w : ndarray; The triangular window, with '
            'the first and last samples equal to zero and the maximum value '
            'normalized to 1 (though the value 1 does not appear if `M` is '
            'even and `sym` is True).'
    },
    'hann': {
        'description': 'Return a Hann window. The Hann window is a taper '
            'formed by using a raised cosine or sine-squared with ends that '
            'touch zero.'
    },
    'tukey': {
        'json_conf': {
            'alpha': {
                'type': 'number',
                'description': 'Shape parameter of the Tukey window, '
                    'representing the fraction of the window inside the '
                    'cosine tapered region. If zero, the Tukey window is '
                    'equivalent to a rectangular window. If one, the Tukey '
                    'window is equivalent to a Hann window.',
            }
        },
        'description': 'Return a Tukey window, also known as a tapered '
            'cosine window.'
    },
    'barthann': {
        'description': 'Return a modified Bartlett-Hann window.'
    },
    'general_hamming': {
        'json_conf': {
            'alpha': {
                'type': 'number',
                'description': 'The window coefficient.',
            }
        },
        'description': 'Return a generalized Hamming window. The generalized '
            'Hamming window is constructed by multiplying a rectangular '
            'window by one period of a cosine function'
    },
    'hamming': {
        'description': 'Return a Hamming window. The Hamming window is a '
            'taper formed by using a raised cosine with non-zero endpoints, '
            'optimized to minimize the nearest side lobe.'
    },
    'kaiser': {
        'json_conf': {
            'beta': {
                'type': 'number',
                'description': 'Shape parameter, determines trade-off between '
                    'main-lobe width and side lobe level. As beta gets large, '
                    'the window narrows.',
            }
        },
        'description': 'Return a Kaiser window. The Kaiser window is a taper '
            'formed by using a Bessel function.'
    },
    'gaussian': {
        'json_conf': {
            'std': {
                'type': 'number',
                'description': 'The standard deviation, sigma.',
            }
        },
        'description': 'Return a Gaussian window.'
    },
    'general_gaussian': {
        'json_conf': {
            'p': {
                'type': 'number',
                'description': 'Shape parameter.  p = 1 is identical to '
                    '`gaussian`, p = 0.5 is the same shape as the Laplace '
                    'distribution.',
            },
            'sig': {
                'type': 'number',
                'description': 'The standard deviation, sigma.',
            }
        },
        'description': 'Return a window with a generalized Gaussian shape.'
    },
    'chebwin': {
        'json_conf': {
            'at ': {
                'type': 'number',
                'description': 'Attenuation (in dB).',
            }
        },
        'description': 'Return a Dolph-Chebyshev window.'
    },
    'cosine': {
        'description': 'Return a window with a simple cosine shape.'
    },
    'exponential': {
        'json_conf': {
            'center': {
                'type': 'number',
                'description': 'Parameter defining the center location of '
                    'the window function. The default value if not given is '
                    '``center = (M-1) / 2``.  This parameter must take its '
                    'default value for symmetric windows.',
            },
            'tau': {
                'type': 'number',
                'description': 'Parameter defining the decay. For '
                    '``center = 0`` use ``tau = -(M-1) / ln(x)`` if ``x`` is '
                    'the fraction of the window remaining at the end.',
            }
        },
        'description': 'Return an exponential (or Poisson) window.'
    },
    'taylor': {
        'json_conf': {
            'nbar': {
                'type': 'integer',
                'description': 'Number of nearly constant level sidelobes '
                    'adjacent to the mainlobe.',
            },
            'sll': {
                'type': 'number',
                'description': 'Desired suppression of sidelobe level in '
                    'decibels (dB) relative to the DC gain of the mainlobe. '
                    'This should be a positive number.',
            },
            'norm': {
                'type': 'boolean',
                'description': 'When True (default), divides the window by '
                    'the largest (middle) value for odd-length windows or the '
                    'value that would occur between the two repeated middle '
                    'values for even-length windows such that all values are '
                    'less than or equal to 1. When False the DC gain will '
                    'remain at 1 (0 dB) and the sidelobes will be `sll` dB '
                    'down.',
                'default': True
            }
        },
        'description': 'Return a Taylor window. The Taylor window taper '
            'function approximates the Dolph-Chebyshev window\'s constant '
            'sidelobe level for a parameterized number of near-in sidelobes, '
            'but then allows a taper beyond . The SAR (synthetic aperature '
            'radar) community commonly uses Taylor weighting for image '
            'formation processing because it provides strong, selectable '
            'sidelobe suppression with minimum broadening of the mainlobe.',
        'desc-return': 'Returns - out : array;  The window. When `norm` is '
            'True (default), the maximum value is normalized to 1 (though '
            'the value 1 does not appear if `M` is even and `sym` is True).'
    },
}


class CusignalWindowNode(TemplateNodeMixin, Node):
    def init(self):
        TemplateNodeMixin.init(self)

        port_type = PortsSpecSchema.port_type
        outports = {'window': {port_type: [cp.ndarray, np.ndarray]}}
        self.template_ports_setup(out_ports=outports)

        meta_outports = {'window': {}}
        self.template_meta_setup(out_ports=meta_outports)

    def conf_schema(self):
        windows_enum = list(_WINS_CONFIG.keys())

        use_cpu_conf = {'use_cpu': {
            'type': 'boolean',
            'description': 'use_cpu - Use CPU for computation via '
                'scipy::signal.windows. Default is False and runs on '
                'GPU via cusignal.',
            'default': False
        }}

        # windows configuration
        win_anyof = []
        for wtype in windows_enum:
            wjson_conf =_DEFAULT_WIN_JSON_CONF.copy()
            wjson_conf_update = _WINS_CONFIG[wtype].get('json_conf', {})
            wjson_conf.update(wjson_conf_update)

            wdesc = '{}\n{}'.format(
                _WINS_CONFIG[wtype]['description'],
                _WINS_CONFIG[wtype].get('desc-return', _DEFAULT_WIN_RETDESC))

            wjson_conf_properties = {
                'window_type': {
                    'type': 'string',
                    'default': wtype,
                    'readOnly': True
                },
                **wjson_conf,
                **use_cpu_conf
            }

            wjson_schema = {
                'title': wtype,
                'description': wdesc,
                'properties': wjson_conf_properties
            }

            win_anyof.append(wjson_schema)

        json = {
            'title': 'Cusignal Correlation Node',
            'type': 'object',
            'default': 'general_cosine',
            'description': 'Filter Window. Parameters updated below based on '
                'selected window.',
            'anyOf': win_anyof,
            'required': ['window_type'],
        }
        return ConfSchema(json=json)

    def process(self, inputs):
        wintype = self.conf.get('window_type', 'general_cosine')
        winmod = siwin if self.conf.get('use_cpu') else cuwin
        winfn = getattr(winmod, wintype)
        # Match function signature parameters from self.conf; apply defaults to
        # anything not matched.
        winsig = inspect.signature(winfn)
        params_filter = [pp.name for pp in winsig.parameters.values()
                         if pp.kind == pp.POSITIONAL_OR_KEYWORD]
        params_dict = {kk: self.conf[kk] for kk in params_filter
                       if kk in self.conf}
        ba = winsig.bind(**params_dict)
        ba.apply_defaults()
        winout = winfn(*ba.args, **ba.kwargs)
        return {'window': winout}
