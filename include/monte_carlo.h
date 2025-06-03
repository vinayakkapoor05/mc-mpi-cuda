#ifndef MONTE_CARLO_H
#define MONTE_CARLO_H

#include <curand_kernel.h>

// initialize each thread’s cuRAND state
__global__ void setup_curand_states(curandState_t *blockStates, unsigned long seed);

// the kernel that does the pi estimation
extern __global__ void pi_estimator_kernel(curandState_t *blockStates, int *block_counts, const long long NUM_POINTS);

#endif // MONTE_CARLO_H
