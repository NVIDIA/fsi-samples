import wave  # Python standard lib.
import struct
try:
    # conda install -c conda-forge pysoundfile
    import soundfile as sf
except ModuleNotFoundError:
    sf = None

import numpy as np
import cupy as cp
import cusignal

from greenflow.dataframe_flow import (
    Node, NodePorts, PortsSpecSchema, ConfSchema, MetaData)
from greenflow.dataframe_flow.template_node_mixin import TemplateNodeMixin

__all__ = ['IQwavefileNode']


def wave_reader(wavefile, nframes):
    '''Read an IQ wavefile. Not thoroughly tested.'''
    # https://stackoverflow.com/questions/19709018/convert-3-byte-stereo-wav-file-to-numpy-array
    with wave.open(wavefile, 'rb') as wf:
        chans = wf.getnchannels()
        # nframes = wf.getnframes()
        sampwidth = wf.getsampwidth()
        if  sampwidth == 3:  # have to read this one sample at a time
            buf = ''
            for _ in range(nframes):
                fr = wf.readframes(1)
                for c in range(0, 3 * chans, 3):
                    # put TRAILING 0 to make 32-bit (file is little-endian)
                    buf += '\0' + fr[c:(c + 3)]
        else:
            buf = wf.readframes(nframes)

    unpstr = '<{0}{1}'.format(nframes * chans,
                              {1:'b', 2:'h', 3:'i', 4:'i', 8:'q'}[sampwidth])
    # x = list(struct.unpack(unpstr, buf))
    wdata = np.array(struct.unpack(unpstr, buf))
    if sampwidth == 3:
        # downshift to get +/- 2^24 with sign extension
        # x = [k >> 8 for k in x]
        wdata = np.right_shift(wdata, 8)

    int2float = 2 ** (sampwidth * 8 - 1) - 1
    # wdata = np.array(x)
    wdata_float = wdata.astype(np.float64) / int2float
    # iq_data = wdata_float.view(dtype=np.complex128)

    return wdata_float


class IQwavefileNode(TemplateNodeMixin, Node):
    def init(self):
        TemplateNodeMixin.init(self)

        outports = {
            'signal': {PortsSpecSchema.port_type: [cp.ndarray, np.ndarray]},
            'framerate': {PortsSpecSchema.port_type: float},
        }
        self.template_ports_setup(out_ports=outports)

        meta_outports = {'signal': {}, 'framerate': {}}
        self.template_meta_setup(out_ports=meta_outports)

    def conf_schema(self):
        json = {
            'title': 'IQ Wavefile Node',
            'type': 'object',
            'description': 'Load IQ data from a *.wav file. Preferably '
                'install "pysoundfile" to do this. Otherwise uses "wave", '  # noqa: E131,E501
                'but it has not been well tested for variety of ways data '
                'has been stored in *.wav files.',
            'properties': {
                'wavefile':  {
                    'type': 'string',
                    'description': 'IQ Wavefile *.wav. Typically '
                        'recorded snippets of SDR IQ.'  # noqa: E131,E501
                },
                'duration': {
                    'type': 'number',
                    'description': 'Number of seconds to load. Number of '
                        'frames loaded is dependent on framerate. Default '  # noqa: E131,E501
                        '1 second. Limited to max frames in file. Will '
                        'fail if exceeds GPU memory size.',
                    'default': 1.0
                },
                'use_cpu': {
                    'type': 'boolean',
                    'description': 'use_cpu - Returns numpy array if True. '
                        'Default is False and returns Cupy array.',  # noqa: E131,E501
                    'default': False
                },
            },
        }
        ui = {'wavefile': {'ui:widget': 'FileSelector'}}
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        infile = self.conf.get('wavefile')
        nsecs = self.conf.get('duration', 1)

        with wave.open(infile) as wf:
            wparams = wf.getparams()
            # buf = wf.readframes(nframes)

        # int2float = (2**15 - 1)
        # wdata = np.frombuffer(buf, dtype=np.int16)
        # wdata_float = wdata.astype(np.float64)/int2float
        # iq_data = wdata_float.view(dtype=np.complex128)

        nframes = min(int(wparams.framerate * nsecs), wparams.nframes)
        if sf is None:
            data = wave_reader(infile, nframes)
            framerate = wparams.framerate
        else:
            data, framerate = sf.read(infile, frames=nframes)

        # IQ data
        cpu_signal = data.view(dtype=np.complex128).reshape(nframes)
        if self.conf.get('use_cpu', False):
            out = {'signal': cpu_signal}
        else:
            # Create mapped, pinned memory for zero copy between CPU and GPU
            gpu_signal_buf = cusignal.get_shared_mem(
                nframes, dtype=np.complex128)
            gpu_signal_buf[:] = cpu_signal

            # zero-copy conversion from Numba CUDA array to CuPy array
            gpu_signal = cp.asarray(gpu_signal_buf)

            out = {'signal': gpu_signal}

        out['framerate'] = float(framerate)

        return out
