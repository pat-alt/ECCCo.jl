#!/bin/bash

#SBATCH --job-name="Fashion MNIST - Grid (ECCCo)"
#SBATCH --time=02:00:00
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=10
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=8GB
#SBATCH --account=research-eemcs-insy
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 

module load 2023r1 openmpi

source experiments/slurm_header.sh

srun julia --project=experiments --threads $SLURM_CPUS_PER_TASK experiments/run_experiments.jl -- data=fmnist output_path=results mpi grid_search threaded n_individuals=10 n_each=32 > experiments/logs/grid_search_fmnist.log