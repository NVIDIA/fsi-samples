##############################################################
# This Dockerfile contains the needed NVIDIA 
# libraries, and plugins to enable the notebook for the GTC 2021
# presentation and course "How to Use GPUs for Accurate Backtesting Equity Investment Straegies"
#
# BUILD: docker build --network=host --file docker/Dockerfile --tag beis:nvidia .
# RUN: docker run --gpus all --rm -it -p 8888:8888 -p 8889:8889 -p 8787:8787 -p 8788:8788 beis:nvidia 
# JUPYTER NOTEBOOK: /jupyter

##############################################################
FROM nvcr.io/nvidia/rapidsai/rapidsai:0.18-cuda11.0-runtime-ubuntu18.04

RUN apt-get update -y && \
    apt-get install -y r-base && \
    apt-get install libcurl4-openssl-dev && which R
RUN R -e "install.packages('tseries',dependencies=TRUE, repos='http://cran.rstudio.com/')"

ADD docker/requirements.txt /workspace/requirements.txt
ADD docker/nvdashboard.json /workspace/nvdashboard.json
ADD ./scripts /rapids/notebooks/FinAnalytics/
RUN cd /rapids/notebooks/FinAnalytics && \
    ln -s /rapids/notebooks/FinAnalytics ../data && \
    sh run.sh
ADD ./LICENSE /rapids/notebooks/
ADD ./README.md /rapids/notebooks/
ADD ./Part1.2021.ipynb /rapids/notebooks/
ADD ./Part2.2021.ipynb /rapids/notebooks/

EXPOSE 8888
EXPOSE 8889
EXPOSE 8787
EXPOSE 8788

WORKDIR /rapids/notebooks
CMD echo You now have 2 quarters of NYSE and NASDAQ prices so notebook runs will be OUT OF SAMPLE as expected.
CMD echo You may modify bottom of scripts/createDirOfPrices.R lines and rerun docker build and docker run to download another pair of quarters.
CMD echo Attempting to start jupyter now...
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''"]
