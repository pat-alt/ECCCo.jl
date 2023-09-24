#!/bin/bash

#SBATCH --job-name="Grid-search MNIST (ECCCo)"
#SBATCH --time=32:00:00
#SBATCH --ntasks=1000
#SBATCH --cpus-per-task=1
#SBATCH --partition=general
#SBATCH --mem-per-cpu=8GB
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 

module use /opt/insy/modulefiles          # Use DAIC INSY software collection
module load openmpi

srun julia --project=experiments experiments/run_experiments.jl -- data=mnist output_path=results mpi grid_search > experiments/grid_search_mnist.log