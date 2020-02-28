FROM nvcr.io/nvidia/pytorch:19.10-py3
USER root

SHELL ["bash","-c"]

#
# Additional python libs
#
RUN pip install cupy-cuda101

RUN conda install -y -c rapidsai -c nvidia -c conda-forge cudf=0.11 cudatoolkit=10.1 dask-cudf=0.11 dask-cuda=0.11 jupyterlab=0.35.4

#RUN conda install -y -c pytorch ignite 
RUN pip install pytorch-ignite

RUN conda install -y nodejs

RUN jupyter labextension install @ijmbarr/jupyterlab_spellchecker

RUN mkdir /.local /.jupyter /.config /.cupy  && chmod 777 /.local /.jupyter /.config /.cupy

# RUN cp -r /usr/lib/python3.6/dist-packages/tensorrt /opt/conda/lib/python3.6/site-packages/tensorrt
# # Add TensorRT executable to path (trtexec)
# ENV PATH=$PATH:/usr/src/tensorrt/bin


# Here's a good place to install pip reqs from JoC repo.
# At the same step, also install TRT pip reqs
WORKDIR /tmp/pipReqs
RUN pip install pycuda pillow
# NeMo toolkit
RUN pip install nemo-toolkit==0.9.0


EXPOSE 8888
EXPOSE 8787
EXPOSE 8786

WORKDIR /
