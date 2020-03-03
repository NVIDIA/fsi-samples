/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cassert>
#include <cstring>
#include <vector>

#include "NvInfer.h"
#include "eluPlugin.h"
#include "pluginKernels.h"
#include "common.h"
#include "logger.h"

using namespace nvinfer1;

namespace elu
{

// constants for approximating the normal cdf
constexpr float A = 0.0;
constexpr float B = 1.0; // sqrt(2.0/M_PI)

template <typename T, unsigned TPB>
__global__ void eluKernel(const T a, const T b, int n, const T* input, T* output)
{

    const int idx = blockIdx.x * TPB + threadIdx.x;

    if (idx < n)
    {
        const T in = input[idx];
        const T tmp = exp(in) - b;
        const T result = (a > in ? a : in) + (a < tmp ? a : tmp);
        output[idx] = result;
    }
}

inline int computeElu(cudaStream_t stream, int n, const float* input, float* output)
{

    constexpr int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;
    eluKernel<float, blockSize><<<gridSize, blockSize, 0, stream>>>(A, B, n, input, output);

    CHECK(cudaPeekAtLastError());
    return 0;
}

inline int computeElu(cudaStream_t stream, int n, const half* input, half* output)
{
    const int blockSize = 256;

    if (0 == (n & 1))
    {
        const int n2 = n / 2;

        const int gridSize = (n2 + blockSize - 1) / blockSize;
        const half2 A2 = __floats2half2_rn(A, A);
        const half2 B2 = __floats2half2_rn(B, B);
        const half2* input2 = reinterpret_cast<const half2*>(input);
        half2* output2 = reinterpret_cast<half2*>(output);
        eluKernel<half2, blockSize><<<gridSize, blockSize, 0, stream>>>(A2, B2, n2, input2, output2);
    }
    else
    {
        const int gridSize = (n + blockSize - 1) / blockSize;
        eluKernel<half, blockSize><<<gridSize, blockSize, 0, stream>>>(A, B, n, input, output);
    }

    CHECK(cudaPeekAtLastError());
    return 0;
}

namespace
{
static const char* GELU_PLUGIN_VERSION{"1"};
static const char* GELU_PLUGIN_NAME{"CustomEluPluginDynamic"};
} // namespace

// Static class fields initialization
PluginFieldCollection EluPluginDynamicCreator::mFC{};
std::vector<PluginField> EluPluginDynamicCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(EluPluginDynamicCreator);

EluPluginDynamic::EluPluginDynamic(const std::string name)
    : mLayerName(name)
{
}

EluPluginDynamic::EluPluginDynamic(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{

    gLogVerbose << "Elu Deser start" << std::endl;
    const char* d = static_cast<const char*>(data);
    const char* a = d;
    mType = readFromBuffer<DataType>(d);
    assert(d == a + length);
    gLogVerbose << "Elu Deser done" << std::endl;
}
// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* EluPluginDynamic::clone() const
{
    return new EluPluginDynamic(mLayerName);
}

nvinfer1::DimsExprs EluPluginDynamic::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder)
{
    return inputs[0];
}

bool EluPluginDynamic::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{

    const PluginTensorDesc& input = inOut[0];
    if (pos == 0)
    {
        return (input.type == DataType::kFLOAT || input.type == DataType::kHALF)
            && (input.format == TensorFormat::kLINEAR);
    }
    if (pos == 1)
    {
        const PluginTensorDesc& output = inOut[1];
        return (input.type == output.type) && (output.format == TensorFormat::kLINEAR);
    }
    return false;
}

void EluPluginDynamic::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs)
{
    mType = in[0].desc.type;
}

size_t EluPluginDynamic::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const
{
    return 0;
}
int EluPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{

    const int inputVolume = samplesCommon::volume(inputDesc[0].dims);
    int status = -1;

    // Our plugin outputs only one tensor
    // Launch CUDA kernel wrapper and save its return value
    if (mType == DataType::kFLOAT)
    {
        const float* input = static_cast<const float*>(inputs[0]);
        float* output = static_cast<float*>(outputs[0]);
        status = computeElu(stream, inputVolume, input, output);
    }
    else if (mType == DataType::kHALF)
    {
        const half* input = static_cast<const half*>(inputs[0]);
        half* output = static_cast<half*>(outputs[0]);
        status = computeElu(stream, inputVolume, input, output);
    }
    else
    {
        assert(false);
    }

    return status;
}

// IPluginV2Ext Methods
nvinfer1::DataType EluPluginDynamic::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    assert(index == 0);
    assert(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF);
    return inputTypes[0];
}

// IPluginV2 Methods

const char* EluPluginDynamic::getPluginType() const
{
    return GELU_PLUGIN_NAME;
}

const char* EluPluginDynamic::getPluginVersion() const
{
    return GELU_PLUGIN_VERSION;
}

int EluPluginDynamic::getNbOutputs() const
{
    return 1;
}

int EluPluginDynamic::initialize()
{
    return 0;
}

void EluPluginDynamic::terminate() {}

size_t EluPluginDynamic::getSerializationSize() const
{
    return sizeof(DataType);
}

void EluPluginDynamic::serialize(void* buffer) const
{
    char *d = static_cast<char*>(buffer), *a = d;
    writeToBuffer(d, mType);
    assert(d == a + getSerializationSize());
}

void EluPluginDynamic::destroy()
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void EluPluginDynamic::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* EluPluginDynamic::getPluginNamespace() const
{
    return mNamespace.c_str();
}

///////////////

EluPluginDynamicCreator::EluPluginDynamicCreator()
{

    // Fill PluginFieldCollection with PluginField arguments metadata
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* EluPluginDynamicCreator::getPluginName() const
{
    return GELU_PLUGIN_NAME;
}

const char* EluPluginDynamicCreator::getPluginVersion() const
{
    return GELU_PLUGIN_VERSION;
}

const PluginFieldCollection* EluPluginDynamicCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* EluPluginDynamicCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    gLogVerbose << "Creating EluPluginDynamic...\n";
    EluPluginDynamic* p = new EluPluginDynamic(name);
    return p;
}

IPluginV2* EluPluginDynamicCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call EluPluginDynamic::destroy()
    return new EluPluginDynamic(name, serialData, serialLength);
}

void EluPluginDynamicCreator::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* EluPluginDynamicCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}
}
