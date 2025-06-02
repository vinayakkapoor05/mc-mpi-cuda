#include <mpi.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "monte_carlo.h"


__global__ void pi_estimator_kernel(curandState_t *states, int *block_counts, const long long NUM_POINTS) {
    extern __shared__ int s_counts[]; // dynamic shared memory (across thread blocks)
    int tid = threadIdx.x;
    // global index of this thread
    long long idx = (long long)blockDim.x * blockIdx.x + tid;


    int local_count = 0;
    if (idx < NUM_POINTS) {
        // produce random point
        float x = curand_uniform(&states[idx]);  
        float y = curand_uniform(&states[idx]);  
        if (x * x + y * y <= 1.0f) {
            local_count = 1;
        }
    }

    // store each thread's local count into shared memory
    s_counts[tid] = local_count;
    __syncthreads(); // synchronization point

    // perform a tree-reduce in shared memory to sum up block-level counts
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_counts[tid] += s_counts[tid + stride];
        }
        __syncthreads();
    }

    // thread 0 of each block writes the block’s total into global memory (vram)
    if (tid == 0) {
        block_counts[blockIdx.x] = s_counts[0];
    }
}
