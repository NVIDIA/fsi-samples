#include <vector>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <curand.h>
 
#define CHECKCURAND(expression)                         \
  {                                                     \
    curandStatus_t status = (expression);                         \
    if (status != CURAND_STATUS_SUCCESS) {                        \
      std::cerr << "Curand Error on line " << __LINE__<< std::endl;     \
      std::exit(EXIT_FAILURE);                                          \
    }                                                                   \
  }

// atomicAdd is introduced for compute capability >=6.0
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
      printf("device arch <=600\n");
        unsigned long long int* address_as_ull = (unsigned long long int*)address;
          unsigned long long int old = *address_as_ull, assumed;
            do {
                    assumed = old;
                        old = atomicCAS(address_as_ull, assumed,
                                                    __double_as_longlong(val + __longlong_as_double(assumed)));
                          } while (assumed != old);
              return __longlong_as_double(old);
}
#endif

__global__ void sumPayoffKernel(float *d_s, const unsigned N_PATHS, double *mysum)
{
  unsigned idx =  threadIdx.x + blockIdx.x * blockDim.x;
  unsigned stride = blockDim.x * gridDim.x;
  unsigned tid = threadIdx.x;

  extern __shared__ double smdata[];
  smdata[tid] = 0.0;

  for (unsigned i = idx; i<N_PATHS; i+=stride)
  {
    smdata[tid] += (double) d_s[i];
  }

  for (unsigned s=blockDim.x/2; s>0; s>>=1)
  {
    __syncthreads();
    if (tid < s) smdata[tid] += smdata[tid + s];
  }

  if (tid == 0)
  {
    atomicAdd(mysum, smdata[0]);
  }
}

__global__ void barrier_option(
    float *d_s,
    const float T,
    const float K,
    const float B,
    const float S0,
    const float sigma,
    const float mu,
    const float r,
    const float * d_normals,
    const long N_STEPS,
    const long N_PATHS)
{
  unsigned idx =  threadIdx.x + blockIdx.x * blockDim.x;
  unsigned stride = blockDim.x * gridDim.x;
  const float tmp1 = mu*T/N_STEPS;
  const float tmp2 = exp(-r*T);
  const float tmp3 = sqrt(T/N_STEPS);
  double running_average = 0.0;

  for (unsigned i = idx; i<N_PATHS; i+=stride)
  {
    float s_curr = S0;
    for(unsigned n = 0; n < N_STEPS; n++){
       s_curr += tmp1 * s_curr + sigma*s_curr*tmp3*d_normals[i + n * N_PATHS];
       running_average += (s_curr - running_average) / (n + 1.0) ;
       if (running_average <= B){
           break;
       }
    }

    float payoff = (running_average>K ? running_average-K : 0.f);
    d_s[i] = tmp2 * payoff;
  }
}

int main(int argc, char *argv[]) {
  try {
    // declare variables and constants
    size_t N_PATHS = 8192000;
    size_t N_STEPS = 365;
    if (argc >= 2)  N_PATHS = atoi(argv[1]);

    if (argc >= 3)  N_STEPS = atoi(argv[2]);

    const float T = 1.0f;
    const float K = 110.0f;
    const float B = 100.0f;
    const float S0 = 120.0f;
    const float sigma = 0.35f;
    const float mu = 0.1f;
    const float r = 0.05f;


    double gpu_sum{0.0};

    int devID{0};
    cudaDeviceProp deviceProps;

    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
    printf("CUDA device [%s]\n", deviceProps.name);
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProps.name, deviceProps.major, deviceProps.minor);
    // Generate random numbers on the device
    curandGenerator_t curandGenerator;
    CHECKCURAND(curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_MTGP32));
    CHECKCURAND(curandSetPseudoRandomGeneratorSeed(curandGenerator, 1234ULL)) ;

    const size_t N_NORMALS = (size_t)N_STEPS * N_PATHS;
    float *d_normals;
    checkCudaErrors(cudaMalloc(&d_normals, N_NORMALS * sizeof(float)));
    CHECKCURAND(curandGenerateNormal(curandGenerator, d_normals, N_NORMALS, 0.0f, 1.0f));
    cudaDeviceSynchronize();

  	// before kernel launch, check the max potential blockSize
  	int BLOCK_SIZE, GRID_SIZE;
  	checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&GRID_SIZE,
  	                                                   &BLOCK_SIZE,
  	                                                   barrier_option,
  	                                                   0, N_PATHS));

  	std::cout << "suggested block size " << BLOCK_SIZE
  	          << " \nsuggested grid size " << GRID_SIZE
  	          << std::endl;

  	std::cout << "Used grid size " << GRID_SIZE << std::endl;

  	// Kernel launch
  	auto t1=std::chrono::high_resolution_clock::now();

  	float *d_s;
  	checkCudaErrors(cudaMalloc(&d_s, N_PATHS*sizeof(float)));

  	auto t3=std::chrono::high_resolution_clock::now();
  	barrier_option<<<GRID_SIZE, BLOCK_SIZE>>>(d_s, T, K, B, S0, sigma, mu, r, d_normals, N_STEPS, N_PATHS);
  	cudaDeviceSynchronize();
  	auto t4=std::chrono::high_resolution_clock::now();

  	double* mySum;
  	checkCudaErrors(cudaMallocManaged(&mySum, sizeof(double)));
  	sumPayoffKernel<<<GRID_SIZE, BLOCK_SIZE, BLOCK_SIZE*sizeof(double)>>>(d_s, N_PATHS, mySum);
  	cudaDeviceSynchronize();
  	auto t5=std::chrono::high_resolution_clock::now();

  	std::cout << "sumPayoffKernel takes "
  	          << std::chrono::duration_cast<std::chrono::microseconds>(t5-t4).count() / 1000.f
  	          << " ms\n";

  	gpu_sum = mySum[0] / N_PATHS;

  	auto t2=std::chrono::high_resolution_clock::now();

  	// clean up
  	CHECKCURAND(curandDestroyGenerator( curandGenerator )) ;
  	checkCudaErrors(cudaFree(d_s));
  	checkCudaErrors(cudaFree(d_normals));
  	checkCudaErrors(cudaFree(mySum));

  	std::cout << "price "
              << gpu_sum
              << " time "
  	          << std::chrono::duration_cast<std::chrono::microseconds>(t5-t1).count() / 1000.f
  	          << " ms\n";
  }

  catch(std::
        exception& e)
  {
    std::cout<< "exception: " << e.what() << "\n";
  }
}
