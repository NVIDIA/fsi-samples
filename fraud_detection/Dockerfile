############################################################################
##
## Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
##
## NVIDIA Sample Code
##
## Please refer to the NVIDIA end user license agreement (EULA) associated
## with this source code for terms and conditions that govern your use of
## this software. Any use, reproduction, disclosure, or distribution of
## this software and related documentation outside the terms of the EULA
## is strictly prohibited.
##
############################################################################
#
# This Dockerfile contains the needed NVIDIA 
# libraries, and plugins to enable the notebooks for FSI Synthetic Data Creation
#
# BUILD: docker build --network=host --file Dockerfile --tag fsi_fd:nvidia .
# RUN: docker run --gpus all --rm -it -p 8888:8888 -p 8889:8889 -p 8787:8787 -p 8788:8788 fsi_fd:nvidia 
# JUPYTER NOTEBOOK: /jupyter

##############################################################

FROM nvcr.io/nvidia/rapidsai/rapidsai:0.19-cuda11.2-runtime-ubuntu20.04

SHELL ["conda", "run", "-n", "rapids", "/bin/bash", "-c"]

# Additional python libs
ADD requirements.txt /
RUN  pip install -r /requirements.txt \
     && apt-get update && apt-get upgrade -y\
     && apt-get install -y p7zip-full

ADD LICENSE /
ADD README.md /
ADD AmlSim /rapids/notebooks/fsi_examples/AmlSim
ADD PaySim /rapids/notebooks/fsi_examples/PaySim
ADD TabFormer /rapids/notebooks/fsi_examples/TabFormer

WORKDIR /rapids/notebooks/fsi_examples

EXPOSE 8888
EXPOSE 8787
EXPOSE 8786
