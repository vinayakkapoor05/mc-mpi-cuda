# Monte Carlo MPI+CUDA for pi estimation
Estimates pi using a distributed Monte Carlo method with MPI + CUDA (+ cuRAND).


## Build

```bash
git clone https://github.com/vinayakkapoor05/monte-carlo-mpi-cuda.git
cd monte-carlo-mpi-cuda
make
```

## Run

### Locally (single node, single GPU)
```bash
mpirun -np 1 ./monte_carlo_mpi_cuda
```

### Cluster with multiple ranks and GPUs
**scripts/run_mpi_cuda_job.sh**  
  Includes SLURM script for running on 4 nodes with 1 GPU each. 

Adjust **run_mpi_cuda_job.sh** as needed
```bash
cd scripts
sbatch run_mpi_cuda_job.sh
```
