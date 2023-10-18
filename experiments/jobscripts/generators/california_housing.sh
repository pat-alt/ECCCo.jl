#!/bin/bash

#SBATCH --job-name="California Housing (ECCCo)"
#SBATCH --time=00:30:00
#SBATCH --ntasks=30
#SBATCH --cpus-per-task=10
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=4GB
#SBATCH --account=research-eemcs-insy
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 

module load 2023r1 openmpi

source experiments/slurm_header.sh

srun julia --project=experiments --threads $SLURM_CPUS_PER_TASK experiments/run_experiments.jl -- data=california_housing output_path=results mpi threaded n_individuals=100 n_runs=5 > experiments/logs/california_housing.log
