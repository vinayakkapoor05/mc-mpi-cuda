#include <mpi.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <iostream>
#include "monte_carlo.h"

int main(int argc, char **argv) {
    // initialize MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto start = std::chrono::high_resolution_clock::now();
    
    // total number of points/trials
    const long long POINTS = 400000000LL;
    const long long NUM_POINTS = POINTS/size;

    // launch config
    const int THREADS_PER_BLOCK = 256;
    int numBlocks = (NUM_POINTS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // allocate device memory
    curandState_t *d_states = nullptr;
    int *d_block_counts = nullptr;
    cudaMalloc(&d_states, NUM_POINTS * sizeof(curandState_t));
    cudaMalloc(&d_block_counts, numBlocks * sizeof(int));

    // launch setup_curand_states
    setup_curand_states<<<numBlocks, THREADS_PER_BLOCK>>>(d_states, 1234ULL, NUM_POINTS);
    cudaDeviceSynchronize(); // synchronization point  


    // launch pi_estimator_kernel
    size_t shared_bytes = THREADS_PER_BLOCK * sizeof(int);
    pi_estimator_kernel<<<numBlocks, THREADS_PER_BLOCK, shared_bytes>>>(d_states, d_block_counts, NUM_POINTS);
    cudaDeviceSynchronize();

    // copy block-level counts
    int *h_block_counts = (int*)malloc(numBlocks * sizeof(int));
    cudaMemcpy(
        h_block_counts,
        d_block_counts,
        numBlocks * sizeof(int),
        cudaMemcpyDeviceToHost
    ); // copy from device to host


    // sum up this rank's count
    long long local_inside = 0;
    for (int i = 0; i < numBlocks; ++i) {
        local_inside += h_block_counts[i];
    } 
    free(h_block_counts);

    // reduce across MPI ranks
    long long global_inside = 0;
    MPI_Reduce(
        &local_inside,
        &global_inside,
        1,
        MPI_LONG_LONG,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0) {
        double pi_estimate = 4.0 * (double)global_inside / (double)(NUM_POINTS * size);
        std::cout << "Pi ≈ " << pi_estimate << std::endl;
    } // only main process (rank == 0) prints the estimate


    // cleanup
    cudaFree(d_states);
    cudaFree(d_block_counts);

    // calculate time-taken
    auto end = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Time Taken: " << elapsed.count() << " seconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
