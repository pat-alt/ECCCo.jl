#!/bin/bash

#SBATCH --job-name="Grid-search California Housing (ECCCo)"
#SBATCH --time=04:00:00
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=8GB
#SBATCH --account=innovation
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 

module load 2023r1 openmpi

srun julia --project=experiments experiments/run_experiments.jl -- data=california_housing output_path=results mpi grid_search n_individuals=20 n_each=32 store_ce > experiments/grid_search_califoenia_housing.log
