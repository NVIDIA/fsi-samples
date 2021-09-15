#!/bin/bash
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

if [ $# -ne 3 ]
then
  echo "Usage: modify_config_param.sh <triton_config_file> <param_name> <param_value>"
fi

triton_config_file=$1
name=$2
value=$3

line_number=`grep \"$name\" -A 2 -n $triton_config_file | tail -n 1 | cut -d "-" -f 1`
cmd="sed -i '${line_number}s/.*/string_value:\"${value}\"/' ${triton_config_file}"
eval $cmd
