#include <mpi.h>
#include "monte_carlo.h"


int main (int argc, char** argv){
    int rank, size;


    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    generate_random_numbers();

    monte_carlo_kernel(rank, size);

    aggregate(rank, size);

    MPI_Finalize();
    return 0;
}


