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
from asr_infer import asr_text
from tts_infer import get_wave
from qa_infer import get_answer


class Model(object):

    def __init__(self):
        pass

    def qa_infer(self, paragraph_text, question):
        return get_answer(paragraph_text, question)

    def asr_infer(self, wav_file):
        return asr_text(wav_file)

    def tacotron_infer(self, text):
        return get_wave(text)
