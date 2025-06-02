#include <mpi.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "monte_carlo.h"

// setup curand states
__global__ void setup_curand_states(curandState_t *states, unsigned long seed, const long long NUM_POINTS) {
    long long idx = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= NUM_POINTS) return;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void pi_estimator_kernel(curandState_t *states, int *block_counts, const long long NUM_POINTS) {
    extern __shared__ int s_counts[];

    int tid = threadIdx.x;
    long long idx = (long long)blockDim.x * blockIdx.x + tid;

    int local_count = 0;
    if (idx < NUM_POINTS) {
        float x = curand_uniform(&states[idx]);  
        float y = curand_uniform(&states[idx]);  
        if (x * x + y * y <= 1.0f) {
            local_count = 1;
        }
    }

    // write into shared memory
    s_counts[tid] = local_count;
    __syncthreads();

    // tree-reduce in shared memory
    // half the threads add pairwise
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_counts[tid] += s_counts[tid + stride];
        }
        __syncthreads();
    }

    // thread 0 of each block writes the block’s total into global memory
    if (tid == 0) {
        block_counts[blockIdx.x] = s_counts[0];
    }
}
