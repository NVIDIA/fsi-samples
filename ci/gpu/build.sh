#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION.
###########################################
# gQuant GPU build and test script for CI #
###########################################
set -e
NUMARGS=$#
ARGS=$*

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Arg parsing function
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Set path and build parallel level
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=4
export CUDA_REL=${CUDA_VERSION%.*}
export CUDA_REL2=${CUDA//./}

# Set home to the job's workspace
export HOME=$WORKSPACE

# Parse git describe
cd $WORKSPACE
export GIT_DESCRIBE_TAG=`git describe --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`

# Enable NumPy's __array_function__ protocol (needed for NumPy 1.16.x,
# will possibly be enabled by default starting on 1.17)
export NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=1

################################################################################
# SETUP - Check environment
################################################################################

logger "Check environment..."
env

logger "Check GPU usage..."
nvidia-smi

logger "Activate conda env..."
source activate gdf
conda list

logger "Install dependencies"
conda install -y "cudf=${RAPIDS_VERSION:-0.10}" "dask-cudf=${RAPIDS_VERSION:-0.10}" networkx "bqplot=0.11.5" xgboost

logger "Check versions..."
python --version
$CC --version
$CXX --version
conda list

################################################################################
# BUILD - Build gQuant
################################################################################

logger "Build gQuant..."
cd $WORKSPACE
python -m pip install -e .


################################################################################
# TEST - Run py.tests for gQuant
################################################################################

if hasArg --skip-tests; then
    logger "Skipping Tests..."
else
    logger "Python py.test for gQuant..."
    cd $WORKSPACE
    py.test -vs --cache-clear --junitxml=${WORKSPACE}/junit-gquant.xml --cov-config=.coveragerc --cov=gquant --cov-report=xml:${WORKSPACE}/gquant-coverage.xml --cov-report term tests/
fi
