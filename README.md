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

## Results
| Trials        | MPI + CUDA Time (s) | Sequential Time (s) | Speedup |
|---------------|---------------------|---------------------|---------|
| 400 million   | 0.81                | 3.47                | 4.3×    |
| 800 million   | 1.36                | 7.16                | 5.3×    |
| 1 billion     | 1.57                | 9.10                | 5.8×    |
| 10 billion    | 12.82               | 99.26               | 7.7×    |

Compute-bound GPU workload: As you increase trials, the GPU runtime scales roughly linearly, indicating minimal MPI/CUDA communication overhead.

## Scaling with Nodes (400 million trials)

| Nodes (GPUs)  | Time (s) | Speedup |
|---------------|---------:|--------:|
| 1 (1 GPU)     | ~3.24    | ~1.1×   |
| 2 (2 GPUs)    | ~1.62    | ~2.1×   |
| 4 (4 GPUs)    | 0.81     | 4.3×    |

Speedup is nearly linear as you go from 1 to 4 GPUs, with a bit of overhead (potentially MPI_Reduce and kernel‐launch latency) showing up on 2 vs. 4 GPUs.
One MPI rank per GPU yields the best throughput, and oversubscribing GPUs (more ranks than devices) degrades performance.


