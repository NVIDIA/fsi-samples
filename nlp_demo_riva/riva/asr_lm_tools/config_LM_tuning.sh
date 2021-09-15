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

# Range of beam width values to consider
beam_width_start=128
beam_width_end=512
beam_width_incrementer=128

# Range of alpha values to consider
alpha_start=0.
alpha_end=3.
alpha_incrementer=0.5

# Range of beta values to consider
beta_start=-2.5
beta_end=1.0
beta_incrementer=0.5

