# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Absolute path to directory which contains the audio files to use for tuning  
# This folder will be mounted inside Docker container
audio_file_dir="/home/user1/wav/test/"

# Absolute path to the ASR manifest json file containing the path to the audio files
# an the corresponding transcripts
# Each line of the .json file must look like: 
#{"audio_filepath": "/home/user1/wav/test/1272-135031-0000.wav","text": "because you were sleeping instead of conquering the lovely rose princess has become a fiddle without a bow while poor shaggy sits there a cooing dove"}
# And example file is provided in the riva-api-client image under /work/wav/test/transcripts.json
audio_file_manifest="/home/user1/wav/test/transcripts.json"

# Range of lm_weight values to consider
lm_weight_start=0.
lm_weight_end=1.0
lm_weight_incrementer=0.1

# Range of word_insertion_score values to consider
word_insertion_score_start=-1.0
word_insertion_score_end=1.0
word_insertion_score_incrementer=0.25

# Range of beam_size values to consider
beam_size_start=16
beam_size_end=64
beam_size_incrementer=16

# Range of beam_size_token values to consider
beam_size_token_start=16
beam_size_token_end=64
beam_size_token_incrementer=16

# Range of beam_threshold values to consider
beam_threshold_start=10.
beam_threshold_end=30.
beam_threshold_incrementer=10.

