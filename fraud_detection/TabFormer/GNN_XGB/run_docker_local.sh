#!/bin/bash

# specify image
IMAGE=pytorch_dgl_rapids

# specify which GPU to use
GPU=$1
# by default select 1st and 2nd GPUs
if [[ -z $GPU ]]; then
     GPU=0,1
fi

GPU2USE='"device=$G"'
GPU2USE=${GPU2USE/\$G/${GPU}}


# Host Post at which Jupyter can be accessed
HOSTPORT=$2
# Port for running Jupyter Lab inside the container
JPORT=$3
USERNAME=`whoami`

# container name
CONTAINER_NAME='dgl_rapids_pytorch_'$USERNAME

# specify default port 8888
if [[ -z  $HOSTPORT ]]; then
    HOSTPORT=8888
fi

# specify default port 8888
if [[ -z  $JPORT ]]; then
    JPORT=8888
fi


docker run -it --rm --gpus=${GPU2USE} \
        -v ${PWD}:/workspace -w /workspace \
	-p ${HOSTPORT}:${JPORT} \
        --name ${CONTAINER_NAME} \
        --shm-size=1g --ulimit memlock=-1 \
	--ulimit stack=6710886 ${IMAGE}
