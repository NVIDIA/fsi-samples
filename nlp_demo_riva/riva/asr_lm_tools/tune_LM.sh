#!/bin/bash
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# This script can be used to find optimal language model hyper-parameters (alpha, beta, beam_width for CPU decoder, and default_beam, lattice_beam, word_insertion_penalty and acoustic_scale for GPU decoder)
# It uses offline recognition without punctuation 

script_path="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <riva_config_file> <LM_tuning_config_file> <decoder_type (cpu|gpu)"
  echo " "  
  echo "   where <riva_config_file> is absolute path to Riva config file (config.sh)"  
  echo "         <LM_tuning_config_file> is absolute path to language model tuning config file (see config_LM_tuning.sh or config_LM_tuning_gpu_decoder.sh)"  
  echo "         <decoder_type> is the type of decoder being used. Can be cpu or gpu"  
  exit 1
fi

riva_config_file=${1}
lm_tuning_config_file=${2}
decoder_type=${3}

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
riva_start_cmd="$script_path/../riva_start.sh ${new_riva_config_file}"

client_cmd="$update_manifest_cmd; /usr/local/bin/riva_asr_client --word_time_offsets=false --riva_uri=localhost:50051 --automatic_punctuation=false --audio_file=/wav/manifest.tmp.json --output_filename=/wav/output.json; python3 /work/utils/calc_wer.py -test /wav/output.json -ref /wav/manifest.tmp.json"
riva_cmd="docker run --init --rm $mnt_args --net host --name ${riva_daemon_client} $image_client /bin/bash -c \"$client_cmd\""

echo "Riva command:"
echo $riva_cmd

now=`date +%Y-%m-%d_%H:%M:%S`
tuning_results_filename="riva_lm_tuning_${now}.csv"

# ---------------
# Start tuning
# ---------------

#Start tuning LM model for CPU decoder
if [[ $decoder_type == "cpu" ]]; then

  triton_config_file=/data/models/ctc-decoder-cpu-streaming-offline/config.pbtxt

  if [[ ! -v beam_width_start || ! -v beam_width_end || ! -v beam_width_incrementer || \
        ! -v alpha_start      || ! -v alpha_end      || ! -v alpha_incrementer      || \
        ! -v beta_start       || ! -v beta_end       || ! -v beta_incrementer         ]]; then
    echo "Make sure the following variables are defined in LM_tuning config file $lm_tuning_config_file:"
    echo ""
    echo "beam_width_start       (current_value = $beam_width_start)"
    echo "beam_width_end         (current_value = $beam_width_end)"
    echo "beam_width_incrementer (current_value = $beam_width_incrementer)"
    echo ""
    echo "alpha_start            (current_value = $alpha_start)"
    echo "alpha_end              (current_value = $alpha_end)"
    echo "alpha_incrementer      (current_value = $alpha_incrementer)"
    echo ""
    echo "beta_start             (current_value = $beta_start)"
    echo "beta_end               (current_value = $beta_end)"
    echo "beta_incrementer       (current_value = $beta_incrementer)"
    echo ""
    exit 1
  fi

  beam_width_arr=$(awk "BEGIN{for(i=$beam_width_start;i<=$beam_width_end;i+=$beam_width_incrementer)print i}")
  alpha_arr=$(awk "BEGIN{for(i=$alpha_start;i<=$alpha_end;i+=$alpha_incrementer)print i}")
  beta_arr=$(awk "BEGIN{for(i=$beta_start;i<=$beta_end;i+=$beta_incrementer)print i}")
  
  echo "beam, alpha, beta, wer" >> $tuning_results_filename
  best_wer=100.
  for beam_width in $beam_width_arr
  do
  for alpha in $alpha_arr
  do
  for beta in $beta_arr
  do
        echo "Terminating Triton and Riva server"
        docker kill $riva_daemon_speech &> /dev/null
  
        # Modify the Triton config file
        docker run --init --rm -v $script_path:/tmp/ -v $riva_model_loc:/data/ ubuntu:18.04 /bin/bash -c "/tmp/modify_config_param.sh ${triton_config_file} beam_search_width ${beam_width}; /tmp/modify_config_param.sh ${triton_config_file} language_model_alpha ${alpha}; /tmp/modify_config_param.sh ${triton_config_file} language_model_beta $beta"
  
        echo "Launching Triton and Riva server"
        eval $riva_start_cmd 
  
        echo "Running ASR with beam_width $beam_width, alpha $alpha, beta $beta"
        echo "riva_cmd: $riva_cmd"
        docker kill riva-asr-client &> /dev/null; eval $riva_cmd &> output_tmp 
        wer_string=$(cat output_tmp | grep -i "Total files" | tr -d $'\r')
        wer=$(echo $wer_string | cut -d ":" -f 3)
        echo "WER: $wer"
        if [ $? -ne 0 ]; then
          echo "Run failed."
        else 
          if (( $(echo "$wer < $best_wer" |bc -l) )); then 
            echo "Updating best result"
            best_wer=$wer
            best_alpha=$alpha
            best_beta=$beta
            best_beam_width=$beam_width
          fi
        fi
        echo "$beam_width, $alpha, $beta, $wer" >> $tuning_results_filename
  done
  done
  done
  
  echo "Best values:"
  echo "Alpha: $best_alpha"
  echo "Beta: $best_beta"
  echo "Beam wdith: $best_beam_width"

#Start tuning LM model for GPU decoder
elif [[ $decoder_type == "gpu" ]]; then

  triton_decoder_config_file=/data/models/ctc-decoder-gpu-streaming-offline/config.pbtxt
  triton_lattice_config_file=/data/models/lattice-post-processor/config.pbtxt

  if [[ ! -v acoustic_scale_start || ! -v acoustic_scale_end || ! -v acoustic_scale_incrementer || \
        ! -v word_insertion_start || ! -v word_insertion_end || ! -v word_insertion_incrementer || \
        ! -v default_beam_start   || ! -v default_beam_end   || ! -v default_beam_incrementer   || \
        ! -v lattice_beam_start   || ! -v lattice_beam_end   || ! -v lattice_beam_incrementer   ]]; then
    echo "Make sure the following variables are defined in LM_tuning config file $lm_tuning_config_file:"
    echo ""
    echo "acoustic_scale_start       (current_value = $acoustic_scale_start)"
    echo "acoustic_scale_end         (current_value = $acoustic_scale_end)"
    echo "acoustic_scale_incrementer (current_value = $acoustic_scale_incrementer)"
    echo ""
    echo "word_insertion_start            (current_value = $word_insertion_start)"
    echo "word_insertion_end              (current_value = $word_insertion_end)"
    echo "word_insertion_incrementer      (current_value = $word_insertion_incrementer)"
    echo ""
    echo "default_beam_start             (current_value = $default_beam_start)"
    echo "default_beam_end               (current_value = $default_beam_end)"
    echo "default_beam_incrementer       (current_value = $default_beam_incrementer)"
    echo ""
    echo "lattice_beam_start             (current_value = $lattice_beam_start)"
    echo "lattice_beam_end               (current_value = $lattice_beam_end)"
    echo "lattice_beam_incrementer       (current_value = $lattice_beam_incrementer)"
    echo ""
    exit 1
  fi

  acoustic_scale_arr=$(awk "BEGIN{for(i=$acoustic_scale_start;i<=$acoustic_scale_end;i+=$acoustic_scale_incrementer)print i}")
  word_insertion_arr=$(awk "BEGIN{for(i=$word_insertion_start;i<=$word_insertion_end;i+=$word_insertion_incrementer)print i}")
  default_beam_arr=$(awk "BEGIN{for(i=$default_beam_start;i<=$default_beam_end;i+=$default_beam_incrementer)print i}")
  lattice_beam_arr=$(awk "BEGIN{for(i=$lattice_beam_start;i<=$lattice_beam_end;i+=$lattice_beam_incrementer)print i}")
  
  echo "default_beam, lattice_beam, word_insertion, acoustic_scale, wer" >> $tuning_results_filename
  best_wer=100.
  for default_beam in $default_beam_arr
  do
  for lattice_beam in $lattice_beam_arr
  do
  for word_insertion in $word_insertion_arr
  do
  for acoustic_scale in $acoustic_scale_arr
  do
        echo "Terminating Triton and Riva server"
        docker kill $riva_daemon_speech &> /dev/null
  
        # Modify the Triton config file
        docker run --init --rm -v $script_path:/tmp/ -v $riva_model_loc:/data/ ubuntu:18.04 /bin/bash -c "/tmp/modify_config_param.sh ${triton_decoder_config_file} default_beam ${default_beam}; /tmp/modify_config_param.sh ${triton_decoder_config_file} lattice_beam ${lattice_beam}; /tmp/modify_config_param.sh ${triton_lattice_config_file} lattice_beam ${lattice_beam}; /tmp/modify_config_param.sh ${triton_lattice_config_file} word_insertion_penalty ${word_insertion}; /tmp/modify_config_param.sh ${triton_decoder_config_file} acoustic_scale ${acoustic_scale}; "
  
        echo "Launching Triton and Riva server"
        eval $riva_start_cmd 
  
        echo "Running ASR with default_beam $default_beam, lattice_beam $lattice_beam, word_insertion $word_insertion, acoustic_scale $acoustic_scale"
        echo "riva_cmd: $riva_cmd"
        docker kill riva-asr-client &> /dev/null; eval $riva_cmd &> output_tmp 
        wer_string=$(cat output_tmp | grep -i "Total files" | tr -d $'\r')
        wer=$(echo $wer_string | cut -d ":" -f 3)
        echo "WER: $wer"
        if [ $? -ne 0 ]; then
          echo "Run failed."
        else 
          if (( $(echo "$wer < $best_wer" |bc -l) )); then 
            echo "Updating best result"
            best_wer=$wer
            best_default_beam=$default_beam
            best_lattice_beam=$lattice_beam
            best_word_insertion=$word_insertion
            best_acoustic_scale=$acoustic_scale
          fi
        fi
        echo "$default_beam, $lattice_beam, $word_insertion, $acoustic_scale, $wer" >> $tuning_results_filename
  done
  done
  done
  done
  
  echo "Best values:"
  echo "Default beam: $best_default_beam"
  echo "Lattice beam: $best_lattice_beam"
  echo "Word insertion: $best_word_insertion"
  echo "Acoustic scale: $best_acoustic_scale"
else
  echo "Invalid value for decoder_type. Must be cpu or gpu"
fi

echo "WER: $best_wer"

#Cleaning up
rm $new_riva_config_file


