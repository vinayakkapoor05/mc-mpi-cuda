#include <curand_kernel.h>
#include "monte_carlo.h"

__global__ void setup_curand_states(curandState_t *blockStates, unsigned long seed) {
    int b = blockIdx.x;
    if (threadIdx.x == 0) {
        curand_init(seed, b, 0, &blockStates[b]);
    }
}
