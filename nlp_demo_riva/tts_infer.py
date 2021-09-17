"""
 ////////////////////////////////////////////////////////////////////////////
 //
 // Copyright (C) NVIDIA Corporation.  All rights reserved.
 //
 // NVIDIA Sample Code
 //
 // Please refer to the NVIDIA end user license agreement (EULA) associated
 // with this source code for terms and conditions that govern your use of
 // this software. Any use, reproduction, disclosure, or distribution of
 // this software and related documentation outside the terms of the EULA
 // is strictly prohibited.
 //
 ////////////////////////////////////////////////////////////////////////////
"""
# TTS proto
import riva_api.riva_tts_pb2 as rtts
import riva_api.riva_tts_pb2_grpc as rtts_srv
import riva_api.riva_audio_pb2 as ra

import grpc
import numpy as np
from wave_utils import add_header

channel = grpc.insecure_channel('riva:50051')
riva_tts = rtts_srv.RivaSpeechSynthesisStub(channel)


def get_wave(text):
    req = rtts.SynthesizeSpeechRequest()
    req.text = text
    # currently required to be "en-US"
    req.language_code = "en-US"
    # Supports LINEAR_PCM, FLAC, MULAW and ALAW audio encodings
    req.encoding = ra.AudioEncoding.LINEAR_PCM
    # ignored, audio returned will be 22.05KHz
    req.sample_rate_hz = 22050
    # ignored
    req.voice_name = "ljspeech"

    resp = riva_tts.Synthesize(req)
    float32_data = np.frombuffer(resp.audio, dtype=np.float32)
    print(float32_data.min(), float32_data.max())
    float32_data = float32_data / 1.414
    float32_data = float32_data * 32767
    int16_data = float32_data.astype(np.int16).tobytes()
    wav = add_header(int16_data, 16, 1, 22050)
    return wav
