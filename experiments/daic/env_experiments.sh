#!/bin/bash

#SBATCH --job-name="/experiments/ environment (ECCCo)"
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=general
#SBATCH --mem-per-cpu=8GB
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 

srun julia --project=experiments experiments/daic/env.jl > experiments/env_experiments.log