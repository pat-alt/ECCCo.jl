#!/bin/bash

#SBATCH --job-name="Linearly Separable (ECCCo)"
#SBATCH --time=01:10:00
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --mem-per-cpu=2GB
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 

module use /opt/insy/modulefiles          # Use DAIC INSY software collection
module load openmpi

source experiments/slurm_header.sh

srun julia --project=experiments --threads $SLURM_CPUS_PER_TASK experiments/run_experiments.jl -- data=linearly_separable output_path=results_testing mpi threaded n_individuals=100 n_runs=2 > experiments/logs/linearly_separable.log
