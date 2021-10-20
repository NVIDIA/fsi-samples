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

# Start from RAPIDS 21.08 image
FROM rapidsai/rapidsai-core:21.08-cuda11.2-base-ubuntu18.04-py3.8
SHELL ["conda", "run", "-n", "rapids", "/bin/bash", "-c"]
# Install additional Python Libraries
ADD requirements.txt /
RUN pip3 install -r /requirements.txt
# Install Pytorch
RUN pip3 install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# Install DGL Nightly Build
RUN pip install --pre dgl-cu111 -f https://data.dgl.ai/wheels-test/repo.html
# Install JupyterLab NVDashboard
RUN pip3 install jupyterlab_nvdashboard

