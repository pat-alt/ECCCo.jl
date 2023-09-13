#!/bin/bash

#SBATCH --job-name="German Credit (ECCCo)"
#SBATCH --time=3:00:00
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=4GB
#SBATCH --account=innovation
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 

module load 2023r1 openmpi

srun julia --project=experiments experiments/run_experiments.jl -- data=german_credit output_path=results retrain mpi > experiments/german_credit.log
