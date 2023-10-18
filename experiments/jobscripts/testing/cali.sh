#!/bin/bash

#SBATCH --job-name="Grid-search California Housing (ECCCo)"
#SBATCH --time=00:35:00
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=5
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-eemcs-insy
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 

module load 2023r1 openmpi

source experiments/slurm_header.sh

srun julia --project=experiments --threads $SLURM_CPUS_PER_TASK experiments/run_experiments.jl -- data=california_housing output_path=results mpi grid_search n_individuals=1 threaded > experiments/logs/grid_search_california_housing.log