#include <mpi.h>
#include <stdio.h>
#include "monte_carlo.h"

extern long long local_inside;
#define NUM_POINTS 100000000

void aggregate(int rank, int size){
    long long global_inside = 0;
    // MPI_Reduce to sum all local_inside and store in global_inside
    MPI_Reduce(&local_inside, &global_inside, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // main process prints final result
    if (rank == 0){
        double pi = 4.0 * (double)global_inside / (double)(NUM_POINTS * size);
        printf("%f\n", pi);
    }
}
