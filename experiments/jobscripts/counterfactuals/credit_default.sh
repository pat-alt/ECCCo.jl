#!/bin/bash

#SBATCH --job-name="Credit Default (ECCCo)"
#SBATCH --time=3:00:00
#SBATCH --ntasks=1000
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=4GB
#SBATCH --account=research-eemcs-insy
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 

module load 2023r1 openmpi

srun julia --project=experiments experiments/run_experiments.jl -- data=credit_default output_path=results mpi > experiments/credit_default.log
