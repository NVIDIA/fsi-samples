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
# NLP proto
import riva_api.riva_nlp_pb2 as rnlp
import riva_api.riva_nlp_pb2_grpc as rnlp_srv
import grpc

channel = grpc.insecure_channel('riva:50051')
riva_nlp = rnlp_srv.RivaLanguageUnderstandingStub(channel)


def get_answer(paragraph_text, question_text):

    total = len(paragraph_text)
    stride = 1024
    final_answer = ''
    final_score = 0
    for i in range(0, total, stride):
        req = rnlp.NaturalQueryRequest()
        req.query = question_text
        req.context = paragraph_text[i:]
        resp = riva_nlp.NaturalQuery(req)
        if resp.results[0].score > final_score and resp.results[0].answer:
            final_answer = resp.results[0].answer
            final_score = resp.results[0].score
    return final_answer
