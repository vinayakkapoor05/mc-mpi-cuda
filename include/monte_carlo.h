#ifndef MONTE_CARLO_H
#define MONTE_CARLO_H

#include <curand_kernel.h>

// initialize each thread’s cuRAND state
__global__ void setup_curand_states(curandState_t *states, unsigned long seed, const long long NUM_POINTS);

extern __global__ void pi_estimator_kernel(curandState_t *states, int *block_counts, const long long NUM_POINTS);

#endif // MONTE_CARLO_H
