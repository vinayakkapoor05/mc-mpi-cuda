#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include 
#include "monte_carlo.h"


static float *h_global_x = nullptr;
static float *h_global_y = nullptr;

static const int  NUM_POINTS = 1000000;

void generate_random_numbers(){
    // only the main process (rank 0) seeds and builds the random number array

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0){
        h_global_x = (float*)malloc(NUM_POINTS * sizeof(float));
        h_global_y = (float*)malloc(NUM_POINTS * sizeof(float));
        srand(time(NULL));

        for (int i = 0; i < NUM_POINTS; i++){
            h_global_x[i] = (float)rand() /  (float)RAND_MAX;
            h_global_y[i] = (float)rand() /  (float)RAND_MAX;

        }
    }
    // main process broadcasts to worker processes
    MPI_Bcast(h_global_x, NUM_POINTS, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_global_y, NUM_POINTS, MPI_FLOAT, 0, MPI_COMM_WORLD);
}