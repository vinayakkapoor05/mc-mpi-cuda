#include <curand_kernel.h>
#include "monte_carlo.h"

// setup curand states
__global__ void setup_curand_states(curandState_t *states, unsigned long seed, const long long NUM_POINTS) {

    // calculate global thread index
    long long idx = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= NUM_POINTS) return;
    // initialize the random number generator so each thread has a unique stream
    curand_init(seed, idx, 0, &states[idx]);
}
