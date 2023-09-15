#!/bin/bash

#SBATCH --job-name="Train German Credit (ECCCo)"
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=4GB
#SBATCH --account=research-eemcs-insy
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 

srun julia --project=experiments experiments/run_experiments.jl -- data=german_credit output_path=results only_models > experiments/train_german_credit.log