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
import grpc
import riva_api.riva_audio_pb2 as ra
import riva_api.riva_asr_pb2 as rasr
import riva_api.riva_asr_pb2_grpc as rasr_srv
import io
import wave


channel = grpc.insecure_channel("riva:50051")
client = rasr_srv.RivaSpeechRecognitionStub(channel)


def asr_text(data):
    audio_data = data.read()
    wf = wave.open(io.BytesIO(audio_data), 'rb')
    rate = wf.getframerate()
    config = rasr.RecognitionConfig(
        encoding=ra.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=rate,
        language_code="en-US",
        max_alternatives=1,
        enable_automatic_punctuation=True,
        audio_channel_count=1,
    )

    request = rasr.RecognizeRequest(config=config, audio=audio_data)

    response = client.Recognize(request)
    print(response)

    if len(response.results[0].alternatives) > 0:
        asr_best_transcript = response.results[0].alternatives[0].transcript
    else:
        asr_best_transcript = ''
    return asr_best_transcript
