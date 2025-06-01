#include <mpi.h>

void generate_random_numbers();
void monte_carlo_gpu_kernel();
void aggregate();

int main (int argc, char** argv[]){
    generate_random_numbers();
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    monte_carlo_gpu_kernel();
    MPI_Finalize();
    aggregate();
    return 0;
}
