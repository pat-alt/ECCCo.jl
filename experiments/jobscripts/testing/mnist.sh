#!/bin/bash

#SBATCH --job-name="MNIST (ECCCo)"
#SBATCH --time=01:00:00
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=4
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=16GB
#SBATCH --account=research-eemcs-insy
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 

module load 2023r1 openmpi

source experiments/slurm_header.sh

srun julia --project=experiments --threads $SLURM_CPUS_PER_TASK experiments/run_experiments.jl -- data=mnist output_path=results mpi threaded n_individuals=48 n_runs=5 n_each=8 > experiments/logs/mnist.log
