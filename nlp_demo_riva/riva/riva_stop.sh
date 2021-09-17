#!/bin/bash
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

check_docker_version() {
    version_string=$(docker version --format '{{.Server.Version}}')
    if [ $? -ne 0 ]; then
        echo "Unable to run Docker. Please check that Docker is installed and functioning."
        exit 1
    fi
    maj_ver=$(echo $version_string | awk -F. '{print $1}')
    min_ver=$(echo $version_string | awk -F. '{print $2}')
    if [ "$maj_ver" -lt "19" ] || ([ "$maj_ver" -eq "19" ] && [ "$min_ver" -lt "03" ]); then
        echo "Docker version insufficient. Please use Docker 19.03 or later"
        exit 1;
    fi
}

# BEGIN SCRIPT
check_docker_version

# load config file
script_path="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
config_path="${script_path}/config.sh"
if [[ ! -f $config_path ]]; then
    echo 'Unable to load configuration file. Override path to file with -c argument.'
    exit 1
fi
source $config_path

echo "Shutting down docker containers..."
docker kill $riva_daemon_speech &> /dev/null
