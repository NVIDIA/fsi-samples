#!/bin/bash
#
# Adopted from https://github.com/tmcdonell/travis-scripts/blob/dfaac280ac2082cd6bcaba3217428347899f2975/update-accelerate-buildbot.sh

set -e

# Setup 'gpuci_retry' for upload retries (results in 4 total attempts)
export GPUCI_RETRY_MAX=3
export GPUCI_RETRY_SLEEP=30

# Set default label options if they are not defined elsewhere
export LABEL_OPTION=${LABEL_OPTION:-"--label main"}

# Skip uploads unless BUILD_MODE == "branch"
if [ ${BUILD_MODE} != "branch" ]; then
  echo "Skipping upload"
  return 0
fi

# Skip uploads if there is no upload key
if [ -z "$MY_UPLOAD_KEY" ]; then
  echo "No upload key"
  return 0
fi

CUDA_REL=${CUDA_VERSION%.*}

SOURCE_BRANCH=master

LABEL_OPTION="--label main"
echo "LABEL_OPTION=${LABEL_OPTION}"

# Restrict uploads to master branch
if [ ${GIT_BRANCH} != ${SOURCE_BRANCH} ]; then
  echo "Skipping upload"
  return 0
fi

################################################################################
# SETUP - Get conda file output locations
################################################################################

gpuci_logger "Get conda file output locations"
export GQAUNT_FILE=`conda build conda/recipes/greenflow --python=$PYTHON --output`

################################################################################
# UPLOAD - Conda packages
################################################################################

if [ "$UPLOAD_GREENFLOW" == "1" ]; then
  test -e ${GQAUNT_FILE}
  echo "Upload greenflow"
  echo ${GQAUNT_FILE}
  gpuci_retry anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --force ${GQAUNT_FILE}
fi
