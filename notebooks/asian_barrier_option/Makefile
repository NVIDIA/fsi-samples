CUDA_HOME ?= /usr/local/cuda/
NVCC ?= nvcc -O3 -DGPUTIMING # -lineinfo

INCLUDES ?= -I$(CUDA_HOME)/include -I$(CUDA_HOME)/samples/common/inc

LIBS ?= -L$(CUDA_HOME)/lib64 -lcudart -lcurand 

NVFLAGS ?= -std=c++11 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 
# Compile cuda source codes to objects
out: cuda_pricing.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(LIBS) -o $@ $<
