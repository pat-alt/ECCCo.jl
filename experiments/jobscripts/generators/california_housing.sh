#!/bin/bash

#SBATCH --job-name="California Housing (ECCCo)"
#SBATCH --time=3:00:00
#SBATCH --ntasks=100
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=8GB
#SBATCH --account=research-eemcs-insy
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 

module load 2023r1 openmpi

srun julia --project=experiments experiments/run_experiments.jl -- data=california_housing output_path=results mpi > experiments/california_housing.log
