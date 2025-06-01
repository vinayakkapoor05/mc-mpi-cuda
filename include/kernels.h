#ifndef KERNELS_H
#define KERNELS_H

__global__ void pi_estimator_kernel(float *x, float *y, int *inside, int num_points);

#endif // KERNELS_H 