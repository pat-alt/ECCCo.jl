#!/bin/bash

#SBATCH --job-name="Grid-search Fashion MNIST (ECCCo)"
#SBATCH --time=32:00:00
#SBATCH --ntasks=1000
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=8GB
#SBATCH --account=research-eemcs-insy
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 

module load 2023r1 openmpi

srun julia --project=experiments experiments/run_experiments.jl -- data=fmnist output_path=results mpi grid_search > experiments/grid_search_fmnist.log