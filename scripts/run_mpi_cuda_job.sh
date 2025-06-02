#!/bin/bash
#SBATCH -A XXXX
#SBATCH -p gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH -N 4
#SBATCH -n 4
#SBATCH -t 1:00:00
#SBATCH --mem=1G
#SBATCH -J monte_carlo_job
#SBATCH -o monte_carlo.out
#SBATCH -e monte_carlo.err

module purge
module load gcc/11.2.0
module load cuda/cuda-12.1.0-openmpi-4.1.4

cd ~/monte-carlo-mpi-cuda

srun ./monte_carlo_mpi_cuda