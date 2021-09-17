# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Absolute path to directory which contains the audio files to use for tuning  
# This folder will be mounted inside Docker container
audio_file_dir="/tuning_data/"

# Absolute path to the ASR manifest json file containing the path to the audio files
# an the corresponding transcripts
# Each line of the .json file must look like: 
#{"audio_filepath": "/home/user1/wav/test/1272-135031-0000.wav","text": "because you were sleeping instead of conquering the lovely rose princess has become a fiddle without a bow while poor shaggy sits there a cooing dove"}
# And example file is provided in the riva-api-client image under /work/wav/test/transcripts.json
# And example file is provided in the riva-api-client image under /work/wav/test/
audio_file_manifest="/tuning_data/transcripts.json"

# Range acoustic_scale values to consider
acoustic_scale_start=1.0
acoustic_scale_end=3.0
acoustic_scale_incrementer=0.25

# Range of word_insertion values to consider
word_insertion_start=4.
word_insertion_end=8.
word_insertion_incrementer=0.5

# Range of default_beam values to consider
default_beam_start=13
default_beam_end=16
default_beam_incrementer=1

# Range of lattice_beam values to consider
lattice_beam_start=3
lattice_beam_end=7
lattice_beam_incrementer=1

