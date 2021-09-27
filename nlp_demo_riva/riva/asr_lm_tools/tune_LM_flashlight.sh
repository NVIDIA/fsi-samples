#!/bin/bash
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# This script can be used to find optimal language model hyper-parameters for flashlight decoder used with Citrinet)
# It uses offline recognition without punctuation 

script_path="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <riva_config_file> <LM_tuning_config_file>"
  echo " "  
  echo "   where <riva_config_file> is absolute path to Riva config file (config.sh)"  
  echo "         <LM_tuning_config_file> is absolute path to language model tuning config file (see config_LM_tuning_flashlight_decoder.sh)"  
  exit 1
fi

riva_config_file=${1}
lm_tuning_config_file=${2}

# Creating new config file with only ASR service enabled
new_riva_config_file=$(mktemp /tmp/riva_config.XXXXXX)
echo "New config file: $new_riva_config_file"
cp $riva_config_file $new_riva_config_file

#Disable all services except ASR
echo "service_enabled_asr=true" >> $new_riva_config_file
echo "service_enabled_nlp=false" >> $new_riva_config_file
echo "service_enabled_tts=false" >> $new_riva_config_file

source $new_riva_config_file
source $lm_tuning_config_file 

if [ ! -d "$audio_file_dir" ]; then
  echo "audio_file_dir $audio_file_dir does not exist. Please update ${lm_tuning_config_file}"
  exit 1
fi

if [ ! -f "$audio_file_manifest" ]; then
  echo "audio_file_manifest $audio_file_manifest does not exist. Please update ${lm_tuning_config_file}"
  exit 1
fi

sed_path=$(echo "${audio_file_dir}" | sed 's/\//\\\//g') 
update_manifest_cmd="sed 's/$sed_path/\/wav\//g' /wav/manifest.json > /wav/manifest.tmp.json"
mnt_args="-v ${audio_file_dir}:/wav/ -v $audio_file_manifest:/wav/manifest.json"

#Launch Riva
riva_start_cmd="bash $script_path/../riva_start.sh ${new_riva_config_file}"

client_cmd="$update_manifest_cmd; /usr/local/bin/riva_streaming_asr_client --num_parallel_requests=128  --chunk_duration_ms=1600  --model_name=citrinet-1024-asr-trt-ensemble-vad-streaming-offline  --interim_results=false  --word_time_offsets=false  --riva_uri=localhost:50051  --automatic_punctuation=false  --audio_file=/wav/manifest.tmp.json  --output_filename=/wav/output.json;  python3 /work/utils/calc_wer.py -test /wav/output.json -ref /wav/manifest.tmp.json"

riva_cmd="docker run --init --rm $mnt_args --net host --name ${riva_daemon_client} $image_client /bin/bash -c \"$client_cmd\""
echo "Riva command:"
echo $riva_cmd

now=`date +%Y-%m-%d_%H:%M:%S`
tuning_results_filename="riva_lm_tuning_${now}.csv"

# ---------------
# Start tuning
# ---------------

#Use offline model to tune, faster
triton_config_file=/data/models/citrinet-1024-asr-trt-ensemble-vad-streaming-offline-ctc-decoder-cpu-streaming-offline/config.pbtxt

lm_weight_arr=$(awk "BEGIN{for(i=$lm_weight_start;i<=$lm_weight_end;i+=$lm_weight_incrementer)print i}")
word_insertion_score_arr=$(awk "BEGIN{for(i=$word_insertion_score_start;i<=$word_insertion_score_end;i+=$word_insertion_score_incrementer)print i}")
beam_size_arr=$(awk "BEGIN{for(i=$beam_size_start;i<=$beam_size_end;i+=$beam_size_incrementer)print i}")
beam_size_token_arr=$(awk "BEGIN{for(i=$beam_size_token_start;i<=$beam_size_token_end;i+=$beam_size_token_incrementer)print i}")
beam_threshold_arr=$(awk "BEGIN{for(i=$beam_threshold_start;i<=$beam_threshold_end;i+=$beam_threshold_incrementer)print i}")

echo "lm_weight, word_insertion_score, beam_size, beam_size_token, beam_threshold, wer" >> $tuning_results_filename
best_wer=100.

for beam_size in $beam_size_arr
do
for beam_size_token in $beam_size_token_arr
do
for beam_threshold in $beam_threshold_arr
do
for lm_weight in $lm_weight_arr
do
for word_insertion_score in $word_insertion_score_arr
do
      echo "Terminating Triton and Riva server"
      docker kill $riva_daemon_speech &> /dev/null

      # Modify the Triton config file
      docker run --init --rm -v $script_path:/tmp/ -v $riva_model_loc:/data/ ubuntu:18.04 /bin/bash -c "bash /tmp/modify_config_param.sh ${triton_config_file} lm_weight ${lm_weight}; bash /tmp/modify_config_param.sh ${triton_config_file} word_insertion_score $word_insertion_score; bash /tmp/modify_config_param.sh ${triton_config_file} beam_size $beam_size; bash /tmp/modify_config_param.sh ${triton_config_file} beam_size_token $beam_size_token; bash /tmp/modify_config_param.sh ${triton_config_file} beam_threshold $beam_threshold;"

      echo "Launching Triton and Riva server"
      eval $riva_start_cmd 

      echo "Running ASR with lm_weight $lm_weight, word_insertion_score $word_insertion_score, beam_size $beam_size, beam_size_token $beam_size_token, beam_threshold $beam_threshold"
      echo "riva_cmd: $riva_cmd"
      docker kill ${riva_daemon_client} &> /dev/null; eval $riva_cmd &> output_tmp 
      wer_string=$(cat output_tmp | grep -i "Total files" | tr -d $'\r')
      wer=$(echo $wer_string | cut -d ":" -f 3)
      echo "WER: $wer"
      if [ $? -ne 0 ]; then
        echo "Run failed."
      else 
        if (( $(echo "$wer < $best_wer" |bc -l) )); then 
          echo "Updating best result"
          best_wer=$wer
          best_lm_weight=$lm_weight
          best_word_insertion_score=$word_insertion_score
          best_beam_size=$beam_size
          best_beam_size_token=$beam_size_token
          best_beam_threshold=$beam_threshold
        fi
      fi
      echo "$lm_weight, $word_insertion_score, $beam_size, $beam_size_token, $beam_threshold, $wer" >> $tuning_results_filename
done
done
done
done
done

echo "Best values:"
echo "lm_weight: $best_lm_weight"
echo "word_insertion_score: $best_word_insertion_score"
echo "beam_size: $best_beam_size"
echo "beam_size_token: $best_beam_size_token"
echo "beam_threshold: $best_beam_threshold"

echo "WER: $best_wer"

#Cleaning up
rm $new_riva_config_file


