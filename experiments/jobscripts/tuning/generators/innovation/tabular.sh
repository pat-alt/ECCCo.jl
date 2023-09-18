#!/bin/bash

#SBATCH --job-name="Grid-search Tabular (ECCCo)"
#SBATCH --time=16:00:00
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=4GB
#SBATCH --account=innovation
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 

module load 2023r1 openmpi

srun julia --project=experiments experiments/run_experiments.jl -- data=gmsc,german_credit,credit_default,california_housing output_path=results mpi grid_search > experiments/grid_search_tabular.log