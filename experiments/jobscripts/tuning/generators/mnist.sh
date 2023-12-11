#!/bin/bash

#SBATCH --job-name="Grid-search MNIST (ECCCo)"
#SBATCH --time=04:00:00
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=10
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=16GB
#SBATCH --account=research-eemcs-insy
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 

module load 2023r1 openmpi

source experiments/slurm_header.sh

srun julia --project=experiments --threads $SLURM_CPUS_PER_TASK experiments/run_experiments.jl -- data=mnist output_path=results mpi grid_search threaded n_individuals=10 n_each=16 > experiments/logs/grid_search_mnist.log