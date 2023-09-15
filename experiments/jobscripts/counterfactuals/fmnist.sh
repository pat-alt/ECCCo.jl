#!/bin/bash

#SBATCH --job-name="Fashion-MNIST (ECCCo)"
#SBATCH --time=24:00:00
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=4
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=4GB
#SBATCH --account=research-eemcs-insy
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 

module load 2023r1 openmpi

srun julia --project=experiments experiments/run_experiments.jl -- data=fmnist retrain output_path=results threaded mpi > experiments/fmnist.log
