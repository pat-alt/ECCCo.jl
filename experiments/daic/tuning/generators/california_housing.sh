#!/bin/bash

#SBATCH --job-name="Grid-search California Housing (ECCCo)"
#SBATCH --time=01:00:00
#SBATCH --ntasks=100
#SBATCH --cpus-per-task=1
#SBATCH --partition=general
#SBATCH --mem-per-cpu=4GB
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 

module use /opt/insy/modulefiles          # Use DAIC INSY software collection
module load openmpi

srun julia --project=experiments experiments/run_experiments.jl -- data=california_housing output_path=results mpi grid_search n_individuals=5 > experiments/grid_search_california_housing.log
