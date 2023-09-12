#!/bin/bash

#SBATCH --job-name="GMSC (ECCCo)"
#SBATCH --time=3:00:00
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=4
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=4GB
#SBATCH --account=innovation
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 

module load 2023r1 openmpi

srun julia --project=experiments --threads 4 experiments/run_experiments.jl -- data=gmsc output_path=results retrain threaded mpi > experiments/gmsc.log