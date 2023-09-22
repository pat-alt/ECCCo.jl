#!/bin/bash

#SBATCH --job-name="Grid-search Tabular (ECCCo)"
#SBATCH --time=04:00:00
#SBATCH --ntasks=2000
#SBATCH --cpus-per-task=1
#SBATCH --partition=general
#SBATCH --mem-per-cpu=4GB
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 

module load 2023r1 openmpi

srun julia --project=experiments experiments/run_experiments.jl -- data=gmsc,german_credit output_path=results mpi grid_search > experiments/grid_search_tabular.log
