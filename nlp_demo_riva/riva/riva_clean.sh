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

delete_docker_volume() {

  # detect if docker volume or local filesystem was used to store models
  if [[ "$(docker volume inspect --format '{{ .Name }}' $1)" == "$1" ]]; then
      echo "Deleting docker volume..."
      read -r -p "Found docker volume '$1'. Delete? [y/N] " response
      if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]
      then
          docker volume rm $1 &> /dev/null
      else
          echo "Skipping..."
      fi
  else
      echo "'$1' is not a Docker volume, or has already been deleted."
      if [ -d $1 ]; then
          echo "Local path '$1' exists. Delete manually, if desired, with:"
          echo "rm -rf $1"
      fi
  fi

}

# BEGIN SCRIPT
check_docker_version

# load config file
script_path="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
if [ -z "$1" ]; then
    config_path="${script_path}/config.sh"
else
    config_path=$(readlink -f $1)
fi
if [[ ! -f $config_path ]]; then
    echo 'Unable to load configuration file. Override path to file with -c argument.'
    exit 1
fi
source $config_path

echo "Cleaning up local Riva installation."

docker kill $riva_daemon_speech &> /dev/null
docker rm -f $riva_daemon_speech &> /dev/null

delete_docker_volume $riva_model_loc
