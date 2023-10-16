#!/bin/bash

#SBATCH --job-name="Grid-search German Credit (ECCCo)"
#SBATCH --time=01:00:00
#SBATCH --ntasks=30
#SBATCH --cpus-per-task=30
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-eemcs-insy
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 

module load 2023r1 openmpi

source experiments/slurm_header.sh

srun julia --project=experiments --threads $SLURM_CPUS_PER_TASK experiments/run_experiments.jl -- data=german_credit output_path=results mpi grid_search threaded n_individuals=100 > experiments/logs/grid_search_german_credit.log
