D_FILE=${D_FILE:='Dockerfile.Rapids'}
D_CONT=${D_CONT:='gquant/gquant:latest'}

echo "Fetching latest version of GQuant project"
git clone --recursive https://github.com/rapidsai/gquant

cat > $D_FILE <<EOF
FROM nvcr.io/nvidia/rapidsai/rapidsai:0.7-cuda10.0-devel-ubuntu18.04-gcc7-py3.6

USER root

ADD gquant /rapids

RUN apt-get update
RUN apt-get install -y libfontconfig1 libxrender1

SHELL ["bash","-c"]
#
# Additional python libs
#
RUN source activate rapids \
    && pip install cython matplotlib networkx nxpd graphviz pudb

RUN cd /rapids && source activate rapids \
    && conda install -c conda-forge bqplot nodejs \
    && conda install -y python-graphviz\
    && conda install -y tqdm \
    && conda install -y pytables \
    && conda install -y -f mkl \
    && conda install -y numpy scipy scikit-learn numexpr 
    ## && conda install -c nvidia -c rapidsai -c numba -c conda-forge -c defaults cudf=0.6 python=3.6 cudatoolkit=10.0
    ## && conda install -c rapidsai cudf \
    ## && conda install -c rapidsai cuml \
    ## && git clone https://github.com/rapidsai/cuml.git

#
# required set up
#
RUN source activate rapids \
    && /conda/envs/rapids/bin/jupyter labextension install @jupyter-widgets/jupyterlab-manager \
    && /conda/envs/rapids/bin/jupyter labextension install bqplot \
    && mkdir /.local        \
    && chmod 777 /.local    \
    && mkdir /.jupyter      \
    && chmod 777 /.jupyter  \
    && mkdir /.config       \
    && chmod 777 /.config   \
    && mkdir /.cupy         \
    && chmod 777 /.cupy

RUN source activate rapids  \
   && pip install dask_labextension \
   && pip install sphinx sphinx_rtd_theme recommonmark numpydoc \
   && /conda/envs/rapids/bin/jupyter labextension install dask-labextension \
   && pip install cupy-cuda100

EXPOSE 8888
EXPOSE 8787
EXPOSE 8786

# the addon for vim editor
# RUN source activate rapids  \
#    && /conda/envs/rapids/bin/jupyter labextension install jupyterlab_vim


WORKDIR /
EOF

docker build -f $D_FILE -t $D_CONT .

