#!/bin/bash

#SBATCH --job-name="MNIST test (ECCCo)"
#SBATCH --time=00:30:00
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=4
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=8GB
#SBATCH --account=research-eemcs-insy
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 

module load 2023r1 openmpi

source experiments/slurm_header.sh

srun julia --project=experiments --threads $SLURM_CPUS_PER_TASK experiments/run_experiments.jl -- data=mnist output_path=results_testing mpi grid_search n_individuals=10 threaded n_each=32 > experiments/logs/grid_search_linearly_separable.log
 