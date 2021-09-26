#!/bin/bash

_basedir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUILDDIR="${_basedir}/build"
TOPDIR="$(dirname $(dirname ${_basedir}))"

GREENFLOWDIR="${TOPDIR}/greenflow"
GREENFLOWLABDIR="${TOPDIR}/greenflowlab"
PLUGINSDIR="${TOPDIR}/gQuant/plugins"


main() {

USERID=$(id -u)
USERGID=$(id -g)

D_FILE=${D_FILE:='Dockerfile.dev'}
echo "Building greenflow container..."

echo -e "\nPlease, select your operating system:\n" \
    "- '1' for Ubuntu 18.04\n" \
    "- '2' for Ubuntu 20.04\n"

read -p "Enter your option and hit return [1]-2: " OPERATING_SYSTEM

OPERATING_SYSTEM=${OPERATING_SYSTEM:-1}
case $OPERATING_SYSTEM in
    1)
        echo "Ubuntu 18.04 selected."
        OS_STR="ubuntu18.04"
        ;;
    *)
        echo "Ubuntu 20.04 selected."
        OS_STR="ubuntu20.04"
        ;;
esac

echo -e "\nPlease, select your CUDA version:\n" \
    "- '1' for cuda 11.0\n" \
    "- '2' for cuda 11.2.2\n"

read -p "Enter your option and hit return [1]-2: " CUDA_VERSION

CUDA_VERSION=${CUDA_VERSION:-1}
case $CUDA_VERSION in
    2)
        echo "CUDA 11.2.2 is selected"
        CUDA_STR="11.2.2"
        ;;
    *)
        echo "CUDA 11.0 is selected"
        CUDA_STR="11.0"
        ;;
esac

RAPIDS_CUDA_VER=$(echo ${CUDA_STR} | sed -E 's/([0-9]+\.[0-9]{1,1})[^ ]*/\1/g')

RAPIDS_VERSION="21.06"

mkdir -p ${BUILDDIR}
cp -r ${GREENFLOWDIR} ${BUILDDIR}
rsync -av --progress ${GREENFLOWLABDIR} ${BUILDDIR} --exclude node_modules 
# cp ${TOPDIR}/gQuant/greenflowrc ${BUILDDIR}
cp ${TOPDIR}/gQuant/README.md ${BUILDDIR}

# cp -r ${PLUGINSDIR} ${BUILDDIR}
mkdir -p "${BUILDDIR}/plugins"
rsync -av --progress "${PLUGINSDIR}/gquant_plugin" "${BUILDDIR}/plugins" \
  --exclude data \
  --exclude .cache \
  --exclude many-small \
  --exclude storage \
  --exclude dask-worker-space \
  --exclude __pycache__

rsync -av --progress "${PLUGINSDIR}/dask_plugin" "${BUILDDIR}/plugins" \
  --exclude data \
  --exclude .cache \
  --exclude many-small \
  --exclude storage \
  --exclude dask-worker-space \
  --exclude __pycache__

rsync -av --progress "${PLUGINSDIR}/hrp_plugin" "${BUILDDIR}/plugins" \
  --exclude data \
  --exclude .cache \
  --exclude many-small \
  --exclude storage \
  --exclude dask-worker-space \
  --exclude __pycache__

rsync -av --progress "${PLUGINSDIR}/cusignal_plugin" "${BUILDDIR}/plugins" \
  --exclude data \
  --exclude .cache \
  --exclude many-small \
  --exclude storage \
  --exclude dask-worker-space \
  --exclude __pycache__

rsync -av --progress "${PLUGINSDIR}/simple_example" "${BUILDDIR}/plugins" \
  --exclude data \
  --exclude .cache \
  --exclude many-small \
  --exclude storage \
  --exclude dask-worker-space \
  --exclude __pycache__

rsync -av --progress "${PLUGINSDIR}/nemo_plugin" "${BUILDDIR}/plugins" \
  --exclude data \
  --exclude .cache \
  --exclude many-small \
  --exclude storage \
  --exclude dask-worker-space \
  --exclude __pycache__

read -p "Enable dev model [y/n]:" DEV_MODE
case $DEV_MODE in
    y)
    echo "Dev mode"
    read -r -d '' INSTALL_GREENFLOW<< EOM
## copy greenflowlab extension
ADD --chown=$USERID:$USERGID ./build /home/quant/greenflow
WORKDIR /home/quant/greenflow
EOM
    MODE_STR="dev"
    ;;
    *)
    echo "Production mode"
    read -r -d '' INSTALL_GREENFLOW<< EOM

WORKDIR /home/quant

ADD --chown=$USERID:$USERGID ./build/README.md /home/quant/README.md

## install greenflow
ADD --chown=$USERID:$USERGID ./build/greenflow /home/quant/greenflow
RUN cd /home/quant/greenflow && pip install .

## install greenflowlab extension
ADD --chown=$USERID:$USERGID ./build/greenflowlab /home/quant/greenflowlab
RUN cd /home/quant/greenflowlab && pip install . && \
    jlpm cache clean && jupyter lab clean

RUN jupyter lab build

## install greenflow plugins
ADD --chown=$USERID:$USERGID ./build/plugins /home/quant/plugins
RUN cd /home/quant/plugins/gquant_plugin && pip install .
RUN cd /home/quant/plugins/dask_plugin && pip install .
RUN cd /home/quant/plugins/hrp_plugin && pip install .
RUN cd /home/quant/plugins/cusignal_plugin && pip install .

WORKDIR /home/quant/plugins/gquant_plugin
ENTRYPOINT MODULEPATH=\$HOME/plugins/gquant_plugin/modules jupyter-lab \
  --allow-root --ip=0.0.0.0 --no-browser --NotebookApp.token='' \
  --ContentsManager.allow_hidden=True \
  --ResourceUseDisplay.track_cpu_percent=True \

EOM
    MODE_STR="prod"
    ;;
esac

greenflow_ver=$(grep version "${GREENFLOWDIR}/setup.py" | sed "s/^.*version='\([^;]*\)'.*/\1/")
CONTAINER="nvidia/cuda:${CUDA_STR}-runtime-${OS_STR}"
D_CONT=${D_CONT:="greenflow/greenflow:${greenflow_ver}-Cuda${RAPIDS_CUDA_VER}_${OS_STR}_Rapids${RAPIDS_VERSION}_${MODE_STR}"}


pushd ${_basedir}

cat > $D_FILE <<EOF
FROM $CONTAINER

EXPOSE 8888
EXPOSE 8787
EXPOSE 8786

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository universe && apt-get update && \
    apt-get install -y --no-install-recommends \
        curl git less net-tools iproute2 vim wget locales-all build-essential \
        apt-utils sshfs libfontconfig1 libxrender1 rsync libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /.local /.jupyter /.config /.cupy \
    && chmod 777 /.local /.jupyter /.config /.cupy

ARG USERNAME=quant
ARG USER_UID=$USERID
ARG USER_GID=$USERGID

# Create the user
RUN groupadd --gid \$USER_GID \$USERNAME && \
    useradd --uid \$USER_UID --gid \$USER_GID -m \$USERNAME && \
    apt-get update && \
    apt-get install -y sudo && \
    echo \$USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/\$USERNAME && \
    chmod 0440 /etc/sudoers.d/\$USERNAME

############ here is done for user greenflow #########
USER \$USERNAME

ENV PATH="/home/quant/miniconda3/bin:\${PATH}"
ENV LC_ALL="en_US.utf8"
ARG PATH="/home/quant/miniconda3/bin:\${PATH}"

WORKDIR /home/quant

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm -f Miniconda3-latest-Linux-x86_64.sh && \
    conda init && \
    pip config set global.cache-dir false && \
    conda install -y mamba -n base -c conda-forge

RUN mamba install -y -c rapidsai -c nvidia -c conda-forge -c defaults \
      rapids=$RAPIDS_VERSION cudatoolkit=$RAPIDS_CUDA_VER python=3.8 && \
    conda clean --all -y

RUN mamba install -y -c conda-forge -c defaults \
      jupyterlab'>=3.0.0' jupyter-packaging'>=0.9.2' jupyterlab-system-monitor \
      nodejs=12.4.0 python-graphviz pydot ruamel.yaml && \
    conda clean --all -y && \
    jlpm cache clean && \
    jupyter lab clean

RUN pip install bqplot==0.12.21 && \
    jlpm cache clean && \
    jupyter lab clean

## install the nvdashboard
# pip install git+https://github.com/rapidsai/jupyterlab-nvdashboard.git@branch-0.6
RUN pip install --upgrade pip && \
    pip install jupyterlab-nvdashboard && \
    jlpm cache clean && \
    jupyter lab clean

## install the dask extension
RUN pip install "dask_labextension>=5.0.0" && \
    jlpm cache clean && \
    jupyter lab clean

$INSTALL_GREENFLOW

EOF

docker build --network=host -f $D_FILE -t $D_CONT .

} # end-of-main


main "$@"

popd

exit
