# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#!/usr/bin/env python

import time
import grpc

import numpy as np
import argparse

import riva_api.riva_audio_pb2 as ra
import riva_api.riva_tts_pb2 as rtts
import riva_api.riva_tts_pb2_grpc as rtts_srv
import wave

import pyaudio


def get_args():
    parser = argparse.ArgumentParser(description="Streaming transcription via Riva AI Services")
    parser.add_argument("--server", default="localhost:50051", type=str, help="URI to GRPC server endpoint")
    parser.add_argument("--voice", type=str, help="voice name to use", default="ljspeech")
    parser.add_argument("-o", "--output", default=None, type=str, help="Output file to write last utterance")
    return parser.parse_args()


def main():
    args = get_args()
    channel = grpc.insecure_channel(args.server)
    tts_client = rtts_srv.RivaSpeechSynthesisStub(channel)
    audio_handle = pyaudio.PyAudio()

    print("Example query:")
    print(
        "  Hello, My name is Linda"
        + ", and I am demonstrating speech synthesis with Riva {@EY2}.I. services, running on NVIDIA {@JH}{@IY1}_{@P}{@IY}_{@Y}{@UW0}s."
    )
    req = rtts.SynthesizeSpeechRequest()
    req.text = "Hello"
    req.language_code = "en-US"
    req.encoding = ra.AudioEncoding.LINEAR_PCM
    req.sample_rate_hz = 22050
    req.voice_name = args.voice

    stream = audio_handle.open(format=pyaudio.paFloat32, channels=1, rate=22050, output=True)
    while True:
        print("Speak: ", end='')
        req.text = str(input())
        if args.output:
            wav = wave.open(args.output, 'wb')
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(req.sample_rate_hz)

        print("Generating audio for request...")
        print(f"  > '{req.text}': ", end='')
        start = time.time()
        resp = tts_client.Synthesize(req)
        stop = time.time()
        print(f"Time to first audio: {(stop-start):.3f}s")
        stream.write(resp.audio)
        if args.output:
            dt = np.float32
            f32_output = (np.frombuffer(resp.audio, dtype=np.float32) * 32767).astype(np.int16)
            wav.writeframesraw(f32_output)
            wav.close()
    stream.stop_stream()
    stream.close()


if __name__ == '__main__':
    main()
