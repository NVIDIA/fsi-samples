#////////////////////////////////////////////////////////////////////////////
#//
#// Copyright (C) NVIDIA Corporation.  All rights reserved.
#//
#// NVIDIA Sample Code
#//
#// Please refer to the NVIDIA end user license agreement (EULA) associated
#// with this source code for terms and conditions that govern your use of
#// this software. Any use, reproduction, disclosure, or distribution of
#// this software and related documentation outside the terms of the EULA
#// is strictly prohibited.
#//
#////////////////////////////////////////////////////////////////////////////

# This Dockerfile contains the needed NVIDIA
# libraries, and plugins to enable the notebook for the developer blog:
# "Explaining and Accelerating Machine Learning for Loan Delinquencies"
# By Mark J. Bennett, John Ashley and Patrick Hogan, with major enhancement by Emanuel Scoullos
#
# BUILD: docker build --network=host --file docker/Dockerfile --tag cdr:nvidia .
# RUN: docker run --gpus all --rm -it -p 8888:8888 -p 8889:8889 -p 8890:8890 -p 8891:8891 -v `pwd`:$HOME cdr:nvidia
# JUPYTER NOTEBOOK: /jupyter

#FROM rapidsai/rapidsai:21.10-cuda11.4-runtime-ubuntu20.04-py3.8
FROM nvcr.io/nvidia/rapidsai/rapidsai:21.06-cuda11.2-runtime-ubuntu20.04

ADD docker/requirements.txt /workspace/
SHELL ["conda", "run", "-n", "rapids", "/bin/bash", "-c"]
WORKDIR /rapids/data

RUN pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html && \ 
    pip install -r /workspace/requirements.txt && \
    conda install -c conda-forge shap --yes

WORKDIR /rapids/notebooks

ADD ./LICENSE /.
ADD ./README.md ./
ADD ./*.ipynb ./
ADD ./*.py ./

EXPOSE 8888
EXPOSE 8889
EXPOSE 8890
EXPOSE 8891
EXPOSE 8787
EXPOSE 8786

CMD echo Attempting to start jupyter now...
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''"]
