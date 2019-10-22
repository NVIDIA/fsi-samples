#!/bin/bash

echo "Building gQuant container..."

echo -e "Please, select the option which better fits your system configuration:\n" \
        " - '1' for Ubuntu 16.04 + cuda 9.2\n" \
        " - '2' for Ubuntu 16.04 + cuda 10.0\n" \
        " - '3' for Ubuntu 18.04 + cuda 9.2\n" \
        " - '4' for Ubuntu 18.04 + cuda 10.0"

read -p "Enter your option and hit return [1]-4: " SYSTEM_CONFIGURATION

SYSTEM_CONFIGURATION=${SYSTEM_CONFIGURATION:-1}
case $SYSTEM_CONFIGURATION in
    2)
        echo "Ubuntu 16.04 + cuda 10.0 selected."
        OS_STR='16.04'
        CONTAINER_VER='10.0'
        CUPY='cupy-cuda100'
        ;;
    3)
        echo "Ubuntu 18.04 + cuda 9.2 selected."
        OS_STR='18.04'
        CONTAINER_VER='9.2'
        CUPY='cupy-cuda92'
        ;;
    4)
        echo "Ubuntu 18.04 + cuda 10.0 selected."
        OS_STR='18.04'
	CONTAINER_VER='10.0'
	CUPY='cupy-cuda100'
        ;;
    *)
        echo "Ubuntu 16.04 + cuda 9.2 selected."
        OS_STR='16.04'
        CONTAINER_VER='9.2'
        CUPY='cupy-cuda92'
        ;;
esac

CONTAINER="nvcr.io/nvidia/rapidsai/rapidsai:0.10-cuda${CONTAINER_VER}-runtime-ubuntu${OS_STR}"

read -p "Would you like to install Vim JupyterLab Extension (optional) [N]/y: " VIM_INSTALL

VIM_INSTALL=${VIM_INSTALL:-N}
if [ "$VIM_INSTALL" = "Y" ] || [ "$VIM_INSTALL" = "y" ]; then
    echo "Vim JupyterLab Extension will be installed."
else
    echo "Vim JupyterLab Extension will not be installed."
fi

D_FILE=${D_FILE:='Dockerfile.Rapids'}
D_CONT=${D_CONT:='gquant/gquant:latest'}

mkdir -p gQuant
cp -r ../gquant ./gQuant
cp -r ../task_example ./gQuant
cp ../setup.cfg ./gQuant
cp ../setup.py ./gQuant
cp ../LICENSE ./gQuant
rsync -av --progress ../notebooks ./gQuant --exclude data --exclude .cache --exclude many-small --exclude storage --exclude dask-worker-space --exclude __pycache__

cat > $D_FILE <<EOF
FROM $CONTAINER
USER root

ADD ./gQuant /rapids/gQuant

RUN apt-get update && apt-get install -y libfontconfig1 libxrender1

SHELL ["bash","-c"]

#
# Additional python libs
#
RUN source activate rapids \ 
    && pip install nxpd $CUPY networkx==1.11

RUN source activate rapids \
    && cd /rapids/gQuant \
    && pip install .

RUN source activate rapids \ 
    && conda install -y -c conda-forge dask-labextension recommonmark numpydoc sphinx_rtd_theme pudb \
    python-graphviz bqplot=0.11.5 nodejs=11.11.0 jupyterlab=0.35.4 ipywidgets=7.4.2 pytables mkl numexpr

#
# required set up
#
RUN source activate rapids \ 
    && jupyter labextension install @jupyter-widgets/jupyterlab-manager@0.38.1 --no-build \
    && jupyter labextension install bqplot@0.4.5 --no-build \
    && mkdir /.local /.jupyter /.config /.cupy  \
    && chmod 777 /.local /.jupyter /.config /.cupy

RUN if [ "$VIM_INSTALL" = "Y" ] || [ "$VIM_INSTALL" = "y" ]; then /conda/envs/rapids/bin/jupyter labextension install jupyterlab_vim@0.10.1 --no-build ; fi

RUN source activate rapids \ 
    && jupyter lab build && jupyter lab clean

EXPOSE 8888
EXPOSE 8787
EXPOSE 8786

WORKDIR /rapids
EOF

docker build -f $D_FILE -t $D_CONT .
