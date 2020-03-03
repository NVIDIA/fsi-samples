# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
--expt-relaxed-constexpr \
--expt-extended-lambda \
-gencode arch=compute_70,code=sm_70 \
-gencode arch=compute_75,code=sm_75 \
-Wno-deprecated-declarations")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

set(ELU_LIBS
    cudart
    cublas
    nvinfer
    nvinfer_plugin
    pthread
    z
)

include_directories(
    ./
    ./log
    ./plugins
    /usr/include/x86_64-linux-gnu
    /usr/local/cuda-10.1/targets/x86_64-linux/include
    /workspace/tensorrt/include
    /workspace/tensorrt/samples/common
    /opt/pytorch/pytorch/third_party/cub
)

link_directories(
    /usr/lib/x86_64-linux-gnu
    /usr/local/cuda-10.1/targets/x86_64-linux/lib
    /workspace/tensorrt/lib
)

add_library(common SHARED
    ./log/logger.cpp
)

add_library(my_plugins SHARED
    plugins/eluPlugin.cu
)

target_link_libraries(my_plugins
    common
    ${ELU_LIBS}
)

target_link_libraries(common
    ${ELU_LIBS}
)

