#!/bin/bash

echo "Building gQuant container..."

echo -e "\nPlease, select your operating system:\n" \
     " - '1' for Ubuntu 16.04\n" \
     " - '2' for Ubuntu 18.04\n" \
     " - '3' for CentOS"

read -p "Enter your option and hit return [1]-3: " OPERATING_SYSTEM

OPERATING_SYSTEM=${OPERATING_SYSTEM:-1}
case $OPERATING_SYSTEM in
    2)
	echo "Ubuntu 18.04 selected."
	OS_STR="ubuntu18.04"
	;;
    3)
	echo "CentOS selected."
	OS_STR="centos7"
	;;
    *)
	echo "Ubuntu 16.04 selected."
	OS_STR="ubuntu16.04"
	;;
esac

echo -e "\nPlease, select your cuda version:\n" \
     " - '1' for cuda 9.2\n" \
     " - '2' for cuda 10.0\n" \
     " - '3' for cuda 10.1.2"

read -p "Enter your option and hit return [1]-3: " CUDA_VERSION

RAPIDS_VERSION="0.13"

CUDA_VERSION=${CUDA_VERSION:-1}
case $CUDA_VERSION in
    2)
	echo "cuda 10.0 selected."
	CONTAINER_VER='10.0'
	CUPY='cupy-cuda100'
	;;
    3)
	echo "cuda 10.1.2 selected."
	CONTAINER_VER='10.1'
	CUPY='cupy-cuda101'
	;;
    *)
	echo "cuda 9.2 selected."
	CONTAINER_VER='9.2'
	CUPY='cupy-cuda92'
	;;
esac

CONTAINER="nvcr.io/nvidia/rapidsai/rapidsai:${RAPIDS_VERSION}-cuda${CONTAINER_VER}-runtime-${OS_STR}"

D_FILE=${D_FILE:='Dockerfile.Rapids'}

mkdir -p gQuant
cp -r ../gquant ./gQuant
cp -r ../task_example ./gQuant
cp ../setup.cfg ./gQuant
cp ../setup.py ./gQuant
cp ../LICENSE ./gQuant
rsync -av --progress ../notebooks ./gQuant --exclude data --exclude .cache --exclude many-small --exclude storage --exclude dask-worker-space --exclude __pycache__

gquant_ver=$(grep version gQuant/setup.py | sed "s/^.*version='\([^;]*\)'.*/\1/")
D_CONT=${D_CONT:="gquant/gquant:${gquant_ver}_${OS_STR}_${CONTAINER_VER}_${RAPIDS_VERSION}"}

cat > $D_FILE <<EOF
FROM $CONTAINER
USER root
ADD ./gQuant /rapids/gQuant
RUN if [ "$OS_STR" = "centos7" ]; then \
        yum install -y fontconfig-devel libXrender ; \
    else \
        apt-get update && apt-get install -y libfontconfig1 libxrender1 ; \
    fi
# RUN apt-get update && apt-get install -y libfontconfig1 libxrender1
SHELL ["bash","-c"]
#
# Additional python libs
#
RUN source activate rapids \ 
    && pip install $CUPY

RUN source activate rapids \
    && cd /rapids/gQuant \
    && pip install .
RUN source activate rapids \ 
    && conda install -y -c conda-forge dask-labextension recommonmark numpydoc sphinx_rtd_theme pudb \
    python-graphviz bqplot=0.11.5 nodejs=11.11.0 jupyterlab=0.35.4 ipywidgets=7.4.2 pytables mkl numexpr \
    pydot
#
# required set up
#
RUN source activate rapids \ 
    && jupyter labextension install @jupyter-widgets/jupyterlab-manager@0.38.1 --no-build \
    && jupyter labextension install bqplot@0.4.5 --no-build \
    && mkdir /.local /.jupyter /.config /.cupy  \
    && chmod 777 /.local /.jupyter /.config /.cupy
RUN source activate rapids \ 
    && jupyter lab build && jupyter lab clean
EXPOSE 8888
EXPOSE 8787
EXPOSE 8786
WORKDIR /rapids
EOF

docker build -f $D_FILE -t $D_CONT .
