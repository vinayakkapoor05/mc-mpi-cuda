#include <mpi.h>
#include "monte_carlo.h"
#include <chrono>
#include <iostream>

int main (int argc, char** argv){
    int rank, size;


    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto start  = std::chrono::high_resolution_clock::now();

    generate_random_numbers();

    monte_carlo_kernel(rank, size);

    aggregate(rank, size);
    auto end  = std::chrono::high_resolution_clock::now();

    MPI_Finalize();

    if (rank == 0){
        std::chrono::duration<double> time_taken = end - start; 
        std::cout << "Time Taken : " << time_taken.count() << " seconds\n";
        }
    return 0;
}


