import wave
import sys
import grpc
import time
import argparse

import riva_api.riva_audio_pb2 as ra
import riva_api.riva_asr_pb2 as rasr
import riva_api.riva_asr_pb2_grpc as rasr_srv


def get_args():
    parser = argparse.ArgumentParser(description="Streaming transcription via Riva AI Services")
    parser.add_argument("--num-clients", default=1, type=int, help="Number of client threads")
    parser.add_argument("--num-iterations", default=1, type=int, help="Number of iterations over the file")
    parser.add_argument(
        "--input-file", required=True, type=str, help="Name of the WAV file with LINEAR_PCM encoding to transcribe"
    )
    parser.add_argument(
        "--simulate-realtime", default=False, action='store_true', help="Option to simulate realtime transcription"
    )
    parser.add_argument(
        "--word-time-offsets", default=False, action='store_true', help="Option to output word timestamps"
    )
    parser.add_argument(
        "--max-alternatives",
        default=1,
        type=int,
        help="Maximum number of alternative transcripts to return (up to limit configured on server)",
    )
    parser.add_argument(
        "--automatic-punctuation",
        default=False,
        action='store_true',
        help="Flag that controls if transcript should be automatically punctuated",
    )
    parser.add_argument("--riva-uri", default="localhost:50051", type=str, help="URI to access Riva server")
    parser.add_argument(
        "--no-verbatim-transcripts",
        default=False,
        action='store_true',
        help="If specified, text inverse normalization will be applied",
    )

    return parser.parse_args()


def print_to_file(responses, output_file, max_alternatives, word_time_offsets):
    start_time = time.time()
    with open(output_file, "w") as f:
        for response in responses:
            if not response.results:
                continue
            partial_transcript = ""
            for result in response.results:
                if result.is_final:
                    for index, alternative in enumerate(result.alternatives):
                        f.write(
                            "Time %.2fs: Transcript %d: %s\n"
                            % (time.time() - start_time, index, alternative.transcript)
                        )

                    if word_time_offsets:
                        f.write("Timestamps:\n")
                        f.write("%-40s %-16s %-16s\n" % ("Word", "Start (ms)", "End (ms)"))
                        for word_info in result.alternatives[0].words:
                            f.write(
                                "%-40s %-16.0f %-16.0f\n" % (word_info.word, word_info.start_time, word_info.end_time)
                            )
                else:
                    transcript = result.alternatives[0].transcript
                    partial_transcript += transcript

            f.write(">>>Time %.2fs: %s\n" % (time.time() - start_time, partial_transcript))


def asr_client(
    id,
    output_file,
    input_file,
    num_iterations,
    simulate_realtime,
    riva_uri,
    max_alternatives,
    automatic_punctuation,
    word_time_offsets,
    verbatim_transcripts,
):

    CHUNK = 1600
    channel = grpc.insecure_channel(riva_uri)
    wf = wave.open(input_file, 'rb')

    frames = wf.getnframes()
    rate = wf.getframerate()
    duration = frames / float(rate)
    if id == 0:
        print("File duration: %.2fs" % duration)

    client = rasr_srv.RivaSpeechRecognitionStub(channel)
    config = rasr.RecognitionConfig(
        encoding=ra.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=wf.getframerate(),
        language_code="en-US",
        max_alternatives=max_alternatives,
        enable_automatic_punctuation=automatic_punctuation,
        enable_word_time_offsets=word_time_offsets,
        verbatim_transcripts=verbatim_transcripts,
    )

    streaming_config = rasr.StreamingRecognitionConfig(config=config, interim_results=True)  # read data

    def generator(w, s, num_iterations, output_file):
        try:
            for i in range(num_iterations):
                w = wave.open(input_file, 'rb')
                start_time = time.time()
                yield rasr.StreamingRecognizeRequest(streaming_config=s)
                num_requests = 0
                while 1:
                    d = w.readframes(CHUNK)
                    if len(d) <= 0:
                        break
                    num_requests += 1
                    if simulate_realtime:
                        time_to_sleep = max(0.0, CHUNK / rate * num_requests - (time.time() - start_time))
                        time.sleep(time_to_sleep)
                    yield rasr.StreamingRecognizeRequest(audio_content=d)
                w.close()
        except Exception as e:
            print(e)

    responses = client.StreamingRecognize(generator(wf, streaming_config, num_iterations, output_file))
    print_to_file(responses, output_file, max_alternatives, word_time_offsets)


from threading import Thread

parser = get_args()

print("Number of clients:", parser.num_clients)
print("Number of iteration:", parser.num_iterations)
print("Input file:", parser.input_file)

threads = []
output_filenames = []
for i in range(parser.num_clients):
    output_filenames.append("output_%d.txt" % i)
    t = Thread(
        target=asr_client,
        args=(
            i,
            output_filenames[-1],
            parser.input_file,
            parser.num_iterations,
            parser.simulate_realtime,
            parser.riva_uri,
            parser.max_alternatives,
            parser.automatic_punctuation,
            parser.word_time_offsets,
            not parser.no_verbatim_transcripts,
        ),
    )
    t.start()
    threads.append(t)

for i, t in enumerate(threads):
    t.join()

print(str(parser.num_clients), "threads done, output written to output_<thread_id>.txt")
