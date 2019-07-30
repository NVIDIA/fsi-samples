D_FILE=${D_FILE:='Dockerfile.Rapids'}
D_CONT=${D_CONT:='gquant/gquant:latest'}

echo "Fetching latest version of GQuant project"
git clone --recursive https://github.com/rapidsai/gquant

cat > $D_FILE <<EOF
FROM nvcr.io/nvidia/rapidsai/rapidsai:0.8-cuda10.0-devel-ubuntu18.04-gcc7-py3.6
USER root

ADD gquant /rapids

RUN apt-get update
RUN apt-get install -y libfontconfig1 libxrender1 graphviz

SHELL ["bash","-c"]
#
# Additional python libs
#
RUN source activate rapids \
    && pip install cython matplotlib networkx nxpd graphviz pudb \

RUN cd /rapids && source activate rapids \
    && conda install -y -c conda-forge bqplot=0.11.5 nodejs=11.11.0 jupyterlab=0.35.4 ipywidgets=7.4.2 \
    && conda install -y tqdm \
    && conda install -y pytables \
    && conda install -y -f mkl \
    && conda install -y numpy scipy scikit-learn numexpr 

#
# required set up
#
RUN source activate rapids \
    && jupyter labextension install @jupyter-widgets/jupyterlab-manager@0.38.1 \
    && jupyter labextension install bqplot@0.4.5 \
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
   && pip install cupy-cuda100

EXPOSE 8888
EXPOSE 8787
EXPOSE 8786

# the addon for vim editor
# RUN source activate rapids  \
#     && /conda/envs/rapids/bin/jupyter labextension install jupyterlab_vim


WORKDIR /
EOF

docker build -f $D_FILE -t $D_CONT .
