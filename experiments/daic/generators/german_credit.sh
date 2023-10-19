#!/bin/bash

#SBATCH --job-name="German Credit (ECCCo)"
#SBATCH --time=01:40:00
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=10
#SBATCH --partition=general
#SBATCH --mem-per-cpu=4GB
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 

module use /opt/insy/modulefiles          # Use DAIC INSY software collection
module load openmpi

source experiments/slurm_header.sh

srun julia --project=experiments --threads $SLURM_CPUS_PER_TASK experiments/run_experiments.jl -- data=german_credit output_path=results mpi threaded n_individuals=100 n_runs=50 > experiments/logs/german_credit.log
