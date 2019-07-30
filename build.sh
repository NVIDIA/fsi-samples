D_FILE=${D_FILE:='Dockerfile.Rapids'}
D_CONT=${D_CONT:='gquant/gquant:latest'}

echo "Fetching latest version of gQuant project"
git clone --recursive https://github.com/rapidsai/gQuant

cat > $D_FILE <<EOF
FROM nvcr.io/nvidia/rapidsai/rapidsai:cuda9.2-runtime-ubuntu16.04
USER root

ADD gQuant /rapids/gQuant

RUN apt-get update && apt-get install -y libfontconfig1 libxrender1

SHELL ["bash","-c"]

#
# Additional python libs
#
RUN pip install nxpd graphviz pudb dask_labextension sphinx sphinx_rtd_theme recommonmark numpydoc cupy-cuda92
RUN conda install -y -c conda-forge python-graphviz bqplot=0.11.5 nodejs=11.11.0 jupyterlab=0.35.4 \
    ipywidgets=7.4.2 pytables mkl numexpr

#
# required set up
#
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager@0.38.1 \
    && jupyter labextension install bqplot@0.4.5 \
    && mkdir /.local /.jupyter /.config /.cupy  \
    && chmod 777 /.local /.jupyter /.config /.cupy

EXPOSE 8888
EXPOSE 8787
EXPOSE 8786

WORKDIR /
EOF

docker build -f $D_FILE -t $D_CONT .
