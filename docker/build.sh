#!/bin/bash

main() {

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
        OS_STR="ubuntu16.04"
        ;;
    2)
        echo "Ubuntu 18.04 selected."
        OS_STR="ubuntu18.04"
        ;;
    *)
        echo "Ubuntu 20.04 selected."
        OS_STR="ubuntu20.04"
        ;;
esac

echo -e "\nPlease, select your CUDA version:\n" \
    "- '1' for cuda 10.0\n" \
    "- '2' for cuda 10.1\n" \
    "- '3' for cuda 10.2\n" \
    "- '4' for cuda 11.0 (minimum requirement for Ubuntu 20.04)\n"

read -p "Enter your option and hit return [1]-3: " CUDA_VERSION

RAPIDS_VERSION="0.14.1"

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
    4)
        echo "CUDA 11.0 is selected"
        CUDA_STR="11.0"
        ;;
    *)
        echo "CUDA 10.0 is selected"
        CUDA_STR="10.0"
        ;;
esac


read -p "Enable dev model [y/n]:" DEV_MODE
case $DEV_MODE in
    y)
	echo "Dev mode"
    read -r -d '' INSTALL_GQUANT<< EOM
## copy gquantlab extension
ADD --chown=$USERID:$USERGID ./gQuant /home/quant/gQuant
WORKDIR /home/quant/gQuant
EOM
    MODE_STR="dev"
	;;
    *)
	echo "Production mode"
    read -r -d '' INSTALL_GQUANT<< EOM
## install gquantlab extension
ADD --chown=$USERID:$USERGID ./gQuant /home/quant/gQuant
RUN pip install .
WORKDIR /home/quant/gQuant/gquantlab
RUN pip install .
RUN jupyter lab build
WORKDIR /home/quant/gQuant
ENTRYPOINT MODULEPATH=\$HOME/gQuant/modules jupyter-lab --allow-root --ip=0.0.0.0 --no-browser --NotebookApp.token=''
EOM
    MODE_STR="prod"
	;;
esac

mkdir -p gQuant
cp -r ../gquant ./gQuant
cp -r ../task_example ./gQuant
cp -r ../modules ./gQuant
cp -r ../taskgraphs ./gQuant
cp ../setup.cfg ./gQuant
cp ../setup.py ./gQuant
cp ../LICENSE ./gQuant
cp ../download_data.sh ./gQuant
cp ../gquantrc ./gQuant
rsync -av --progress ../notebooks ./gQuant --exclude data --exclude .cache --exclude many-small --exclude storage --exclude dask-worker-space --exclude __pycache__
rsync -av --progress ../gquantlab ./gQuant --exclude node_modules 

gquant_ver=$(grep version gQuant/setup.py | sed "s/^.*version='\([^;]*\)'.*/\1/")
CONTAINER="nvidia/cuda:${CUDA_STR}-runtime-${OS_STR}"
D_CONT=${D_CONT:="gquant/gquant:${gquant_ver}-${CUDA_STR}_${OS_STR}_${RAPIDS_VERSION}_${MODE_STR}"}


gen_nemo_patches


cat > $D_FILE <<EOF
FROM $CONTAINER
EXPOSE 8888
EXPOSE 8787
EXPOSE 8786
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository universe && apt-get update && \
    apt-get install -y --no-install-recommends \
        curl git net-tools iproute2 vim wget locales-all build-essential \
        libfontconfig1 libxrender1 rsync libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

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
    -c defaults rapids=$RAPIDS_VERSION cudatoolkit=$CUDA_STR python=3.7

RUN conda install -y -c conda-forge jupyterlab 

RUN conda install -y -c conda-forge python-graphviz bqplot nodejs ipywidgets \
    pytables mkl numexpr pydot flask pylint flake8 autopep8

RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build  
RUN jupyter labextension install bqplot --no-build  
#RUN jupyter labextension install jupyterlab-nvdashboard --no-build  
RUN jupyter lab build && jupyter lab clean
RUN conda init

## install the nvdashboard
RUN pip install jupyterlab-nvdashboard
RUN jupyter labextension install jupyterlab-nvdashboard

## install the dask extension
RUN pip install dask_labextension
RUN jupyter labextension install dask-labextension
RUN jupyter serverextension enable dask_labextension

## install the jsonpath lib
RUN pip install jsonpath-ng ray[tune] Cython

## install the NemO
WORKDIR /home/quant/
RUN git clone -b v0.11.1 https://github.com/NVIDIA/NeMo.git 
WORKDIR /home/quant/NeMo
RUN sed -i 's/numba<=0.48/numba==0.49.1/g' requirements/requirements_asr.txt
COPY sacrebleu.patch /home/quant/NeMo/
COPY nemo_paddpatch.patch /home/quant/NeMo/
RUN patch -u nemo/collections/nlp/metrics/sacrebleu.py -i sacrebleu.patch && \
    git apply nemo_paddpatch.patch && \
    bash reinstall.sh

RUN conda install -y ruamel.yaml
RUN conda install -c conda-forge -y cloudpickle

RUN mkdir -p /home/quant/gQuant
WORKDIR /home/quant/gQuant

RUN pip install streamz && \
    pip uninstall -y dataclasses

$INSTALL_GQUANT


EOF
docker build --network=host -f $D_FILE -t $D_CONT .

if [ -f "sacrebleu.patch" ] ; then
    rm sacrebleu.patch
fi

if [ -f "nemo_paddpatch.patch" ] ; then
    rm nemo_paddpatch.patch
fi

} # end-of-main

gen_nemo_patches() {

cat << 'EOF' > sacrebleu.patch
--- nemo/collections/nlp/metrics/sacrebleu.py
+++ sacrebleu_fix.py
@@ -61,13 +61,16 @@
 VERSION = '1.3.5'
 
 try:
+    import threading
     # SIGPIPE is not available on Windows machines, throwing an exception.
     from signal import SIGPIPE
 
     # If SIGPIPE is available, change behaviour to default instead of ignore.
     from signal import signal, SIG_DFL
 
-    signal(SIGPIPE, SIG_DFL)
+
+    if threading.current_thread() == threading.main_thread():
+        signal(SIGPIPE, SIG_DFL)
 
 except ImportError:
     logging.warning('Could not import signal.SIGPIPE (this is expected on Windows machines)')
EOF


cat << 'EOF' > nemo_paddpatch.patch

diff --git a/nemo/backends/pytorch/common/rnn.py b/nemo/backends/pytorch/common/rnn.py
index c1c62ac..b9936fe 100644
--- a/nemo/backends/pytorch/common/rnn.py
+++ b/nemo/backends/pytorch/common/rnn.py
@@ -235,7 +235,7 @@ class EncoderRNN(TrainableNM):
         embedded = self.embedding(inputs)
         embedded = self.dropout(embedded)
         if input_lens is not None:
-            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lens, batch_first=True)
+            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lens.cpu(), batch_first=True)
 
         outputs, hidden = self.rnn(embedded)
         # outputs of shape (seq_len, batch, num_directions * hidden_size)
diff --git a/nemo/backends/pytorch/tutorials/chatbot/modules.py b/nemo/backends/pytorch/tutorials/chatbot/modules.py
index 2459afa..59b88d2 100644
--- a/nemo/backends/pytorch/tutorials/chatbot/modules.py
+++ b/nemo/backends/pytorch/tutorials/chatbot/modules.py
@@ -122,7 +122,7 @@ class EncoderRNN(TrainableNM):
         embedded = self.embedding(input_seq)
         embedded = self.embedding_dropout(embedded)
         # Pack padded batch of sequences for RNN module
-        packed = t.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
+        packed = t.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths.cpu())
         # Forward pass through GRU
         outputs, hidden = self.gru(packed, hidden)
         # Unpack padding
diff --git a/nemo/collections/nlp/nm/trainables/common/encoder_rnn.py b/nemo/collections/nlp/nm/trainables/common/encoder_rnn.py
index 2fc2ff0..9ec7acc 100644
--- a/nemo/collections/nlp/nm/trainables/common/encoder_rnn.py
+++ b/nemo/collections/nlp/nm/trainables/common/encoder_rnn.py
@@ -64,7 +64,7 @@ class EncoderRNN(TrainableNM):
         embedded = self.embedding(inputs)
         embedded = self.dropout(embedded)
         if input_lens is not None:
-            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lens, batch_first=True)
+            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lens.cpu(), batch_first=True)
 
         outputs, hidden = self.rnn(embedded)
         # outputs of shape (seq_len, batch, num_directions * hidden_size)
diff --git a/nemo/collections/tts/parts/tacotron2.py b/nemo/collections/tts/parts/tacotron2.py
index 925251f..5f81647 100644
--- a/nemo/collections/tts/parts/tacotron2.py
+++ b/nemo/collections/tts/parts/tacotron2.py
@@ -221,7 +221,7 @@ class Encoder(nn.Module):
 
         # pytorch tensor are not reversible, hence the conversion
         input_lengths = input_lengths.cpu().numpy()
-        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True, enforce_sorted=False)
+        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
 
         self.lstm.flatten_parameters()
         outputs, _ = self.lstm(x)

EOF

}


main "$@"

exit
