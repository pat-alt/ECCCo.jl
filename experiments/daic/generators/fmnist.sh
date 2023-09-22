#!/bin/bash

#SBATCH --job-name="Fashion-MNIST (ECCCo)"
#SBATCH --time=10:00:00
#SBATCH --ntasks=1000
#SBATCH --cpus-per-task=1
#SBATCH --partition=general
#SBATCH --mem-per-cpu=8GB
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 

module load 2023r1 openmpi

srun julia --project=experiments experiments/run_experiments.jl -- data=fmnist retrain output_path=results threaded mpi > experiments/fmnist.log
