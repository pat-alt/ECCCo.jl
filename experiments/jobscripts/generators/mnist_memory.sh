#!/bin/bash

#SBATCH --job-name="MNIST (ECCCo)"
#SBATCH --time=10:00:00
#SBATCH --ntasks=100
#SBATCH --cpus-per-task=1
#SBATCH --partition=memory
#SBATCH --mem-per-cpu=32GB
#SBATCH --account=research-eemcs-insy
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 

module load 2023r1 openmpi

srun julia --project=experiments experiments/run_experiments.jl -- data=mnist output_path=results mpi > experiments/mnist.log
