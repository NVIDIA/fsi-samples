FROM nvidia/cuda:11.2.1-devel-ubuntu20.04
RUN apt-get update && \ 
    apt-get install -y libfontconfig1 libxrender1 wget && \
    rm -rf /var/lib/apt/lists/*
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \ 
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
ENV PATH="/root/miniconda3/bin:${PATH}"
RUN conda init
RUN conda install -y -c conda-forge python-graphviz nodejs=12.4.0  pydot
RUN conda install -y -c rapidsai -c nvidia -c conda-forge rapids=21.06 python=3.8 cudatoolkit=11.0
RUN conda install -y -c conda-forge ruamel.yaml
RUN pip install greenflow
RUN pip install greenflowlab
RUN pip install greenflow_dask_plugin
RUN pip install greenflow_gquant_plugin
WORKDIR /workspace
COPY greenflow_hrp_plugin greenflow_hrp_plugin 
COPY setup.py .
RUN pip install .
COPY notebooks notebooks 
ENTRYPOINT jupyter-lab --allow-root --ip=0.0.0.0 --no-browser --NotebookApp.token='' 





