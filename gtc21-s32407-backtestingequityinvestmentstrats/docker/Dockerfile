##############################################################
# This Dockerfile contains the needed NVIDIA 
# libraries, and plugins to enable the notebook for the GTC 2021
# presentation and course "How to Use GPUs for Accurate Backtesting Equity Investment Straegies"
#
# BUILD: docker build --network=host --file docker/Dockerfile --tag beis:nvidia .
# RUN: docker run --gpus all --rm -it -p 8888:8888 -p 8889:8889 -p 8787:8787 -p 8786:8788 beis:nvidia 
# JUPYTER NOTEBOOK: /jupyter
##############################################################
FROM nvcr.io/nvidia/rapidsai/rapidsai:0.18-cuda11.0-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive

# Install Python, NodeJS, and Jupyter Lab with extensions
RUN apt-get update -y && \
    apt-get install -y -q \
        apt-utils \
        vim \
        curl && \    
    curl -sL https://deb.nodesource.com/setup_12.x | bash - && \
    apt-get install -y -q \
        nodejs && \
    apt-get update -y && \
    rm -rf /rapids/notebooks && \
    rm -rf /var/lib/apt/lists/* 


ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN ls -ltr;pwd
ADD docker/run-jupyterlab.sh /jupyter
ADD docker/requirements.txt /workspace/requirements.txt
ADD docker/nvdashboard.json /workspace/nvdashboard.json
ADD ./LICENSE /rapids/notebooks/
ADD ./README.md /rapids/notebooks/
ADD ./scripts /rapids/notebooks/FinAnalytics/
ADD ./Part1.2021.ipynb /rapids/notebooks/
ADD ./Part2.2021.ipynb /rapids/notebooks/
    
RUN apt-get update -y && \
    apt-get install -y \
        wget vim git unzip \
        libcurl4 libcurl4-openssl-dev && \
    python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir wheel && \
    python3 -m pip install --no-cache-dir -r /workspace/requirements.txt && \
    rm -rf ~/.cache/pip

RUN chmod +x  /jupyter && \
    /opt/conda/envs/rapids/bin/jupyter labextension install jupyterlab-nvdashboard && \
    /opt/conda/envs/rapids/bin/jupyter labextension install dask-labextension && \
    /opt/conda/envs/rapids/bin/jupyter serverextension enable dask_labextension && \
    /opt/conda/envs/rapids/bin/jupyter lab workspaces import /workspace/nvdashboard.json

RUN apt-get update -y && \
    apt-get install -y r-base   
RUN R -e "install.packages('tseries',dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN file="$(ls -1trF /rapids/notebooks/FinAnalytics)" && echo $file 

RUN cd /rapids/notebooks/FinAnalytics && \
    ln -s /rapids/notebooks/FinAnalytics ../data && \
    sh run.sh 

WORKDIR /rapids/notebooks

EXPOSE 8888
RUN echo You now have 2 quarters of NYSE and NASDAQ prices so notebook runs will be OUT OF SAMPLE as expected. You may modify bottom of scripts/createDirOfPrices.R lines and rerun docker build and docker run to download another pair of quarters.
RUN echo Attempting to start jupyter now...
CMD ["/jupyter"]
