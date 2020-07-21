#!/bin/bash

USERID=$(id -u)
USERGID=$(id -g)

D_FILE=${D_FILE:='Dockerfile.dev'}
echo "Building gQuant container..."

echo -e "\nPlease, select your operating system:\n" \
    "- '1' for Ubuntu 16.04\n" \
    "- '2' for Ubuntu 18.04\n" \
    "- '3' for Ubuntu 20.04\n"

read -p "Enter your option and hit return [1]-3: " OPERATING_SYSTEM

OPERATING_SYSTEM=${OPERATING_SYSTEM:-1}
case $OPERATING_SYSTEM in
    1)
        echo "Ubuntu 16.04 selected."
        OS_STR="10.2-cudnn8-devel-ubuntu16.04"
        ;;
    2)
        echo "Ubuntu 18.04 selected."
        OS_STR="10.2-cudnn8-devel-ubuntu18.04"
        ;;
    *)
        echo "Ubuntu 20.04 selected."
        OS_STR="11.0-devel-ubuntu20.04"
        ;;
esac

echo -e "\nPlease, select your CUDA version:\n" \
    "- '1' for cuda 10.0\n" \
    "- '2' for cuda 10.1\n" \
    "- '3' for cuda 10.2\n"

read -p "Enter your option and hit return [1]-3: " CUDA_VERSION

RAPIDS_VERSION="0.14"

CUDA_VERSION=${CUDA_VERSION:-1}
case $CUDA_VERSION in
    2)
        echo "CUDA 10.1 is selected"
        CUDA_STR="10.1"
        ;;
    3)
        echo "CUDA 10.2 is selected"
        CUDA_STR="10.2"
        ;;
    *)
        echo "CUDA 10.0 is selected"
        CUDA_STR="10.0"
        ;;
esac

D_CONT=${D_CONT:="gquant/gquant:${CUDA_STR}_${OS_STR}_${RAPIDS_VERSION}"}

cat > $D_FILE <<EOF
FROM nvidia/cuda:$OS_STR
EXPOSE 8888
EXPOSE 8787
EXPOSE 8786
RUN apt-get update
RUN apt-get install -y curl git net-tools iproute2 vim wget locales-all build-essential libfontconfig1 libxrender1 \
        && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://code-server.dev/install.sh | sh 
RUN mkdir /.local /.jupyter /.config /.cupy \
    && chmod 777 /.local /.jupyter /.config /.cupy

ARG USERNAME=quant
ARG USER_UID=$USERID
ARG USER_GID=$USERGID


# Create the user
RUN groupadd --gid \$USER_GID \$USERNAME     && useradd --uid \$USER_UID --gid \$USER_GID -m \$USERNAME     && apt-get update     && apt-get install -y sudo     && echo \$USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/\$USERNAME     && chmod 0440 /etc/sudoers.d/\$USERNAME

############ here is done for user gquant #########
USER \$USERNAME

ENV PATH="/home/quant/miniconda3/bin:\${PATH}"
ENV LC_ALL="en_US.utf8"
ARG PATH="/home/quant/miniconda3/bin:\${PATH}"

WORKDIR /home/quant

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && conda init

RUN conda install -y -c rapidsai -c nvidia -c conda-forge \
    -c defaults rapids=$RAPIDS_VERSION python=3.7 cudatoolkit=$CUDA_STR

RUN conda install -y -c conda-forge jupyterlab

RUN conda install -y -c conda-forge python-graphviz bqplot nodejs ipywidgets \
    pytables mkl numexpr pydot flask pylint flake8 autopep8

RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build  
RUN jupyter labextension install bqplot --no-build  
#RUN jupyter labextension install jupyterlab-nvdashboard --no-build  
RUN jupyter lab build && jupyter lab clean
RUN mkdir -p /home/quant/.local/share/code-server/extensions
COPY settings.json /home/quant/.local/share/code-server/User/settings.json
RUN code-server --install-extension ms-python.python@2020.5.86806
RUN code-server --install-extension vscodevim.vim
RUN code-server --install-extension firefox-devtools.vscode-firefox-debug
RUN code-server --install-extension dbaeumer.vscode-eslint
RUN code-server --install-extension esbenp.prettier-vscode
RUN conda init
RUN sudo chmod 777 -R /home/quant/.local/share/
ENTRYPOINT code-server --bind-addr=0.0.0.0:8080 --auth none
EOF

docker build -f $D_FILE -t $D_CONT .
