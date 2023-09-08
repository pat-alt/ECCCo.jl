#!/bin/bash

#SBATCH --job-name="ECCCo"
#SBATCH --time=01:00:00
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=4
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=4GB
#SBATCH --account=innovation
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 

module load 2023r1 openmpi julia

srun julia --project=experiments experiments/run_experiments.jl -- data=linearly_separable threaded retrain > hpc.log
