#!/bin/bash

echo "Building gQuant container..."

read -p "Please, press '1' for cuda 9.2, or '2' for cuda 10.0. [1]/2: " CUDA_VERSION
CUDA_VERSION=${CUDA_VERSION:-1}

if [ "$CUDA_VERSION" -eq 2 ]; then
    echo "cuda 10.0 selected."
    CONTAINER='nvcr.io/nvidia/rapidsai/rapidsai:cuda10.0-runtime-ubuntu16.04'
    CUPY='cupy-cuda100'
else
    echo "cuda 9.2 selected."
    CONTAINER='nvcr.io/nvidia/rapidsai/rapidsai:cuda9.2-runtime-ubuntu16.04'
    CUPY='cupy-cuda92'
fi

read -p "Would you like to install Vim JupyterLab Extension (optional) [N]/y: " VIM_INSTALL
VIM_INSTALL=${VIM_INSTALL:-N}

if [ "$VIM_INSTALL" = "Y" ] || [ "$VIM_INSTALL" = "y" ]; then
    echo "Vim JupyterLab Extension will be installed."
else
    echo "Vim JupyterLab Extension will not be installed."
fi

D_FILE=${D_FILE:='Dockerfile.Rapids'}
D_CONT=${D_CONT:='gquant/gquant:latest'}

echo "Fetching latest version of gQuant project"
git clone --recursive https://github.com/rapidsai/gQuant


cat > $D_FILE <<EOF
FROM $CONTAINER
USER root

ADD gQuant /rapids/gQuant

RUN apt-get update && apt-get install -y libfontconfig1 libxrender1

SHELL ["bash","-c"]

#
# Additional python libs
#
RUN pip install nxpd graphviz pudb dask_labextension sphinx sphinx_rtd_theme recommonmark numpydoc $CUPY
RUN conda install -y -c conda-forge python-graphviz bqplot=0.11.5 nodejs=11.11.0 jupyterlab=0.35.4 \
    ipywidgets=7.4.2 pytables mkl numexpr

#
# required set up
#
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager@0.38.1 \
    && jupyter labextension install bqplot@0.4.5 \
    && mkdir /.local /.jupyter /.config /.cupy  \
    && chmod 777 /.local /.jupyter /.config /.cupy

RUN if [ "$VIM_INSTALL" = "Y" ] || [ "$VIM_INSTALL" = "y" ]; then /conda/envs/rapids/bin/jupyter labextension install jupyterlab_vim@0.10.1 ; fi

EXPOSE 8888
EXPOSE 8787
EXPOSE 8786

WORKDIR /
EOF

docker build -f $D_FILE -t $D_CONT .
