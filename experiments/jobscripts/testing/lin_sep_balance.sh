#!/bin/bash

#SBATCH --job-name="Grid-search Linearly Separable (ECCCo)"
#SBATCH --time=00:30:00
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=5
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-eemcs-insy
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 

module load 2023r1 openmpi

source experiments/slurm_header.sh

srun julia --project=experiments --threads $SLURM_CPUS_PER_TASK experiments/run_experiments.jl -- data=linearly_separable output_path=results mpi grid_search n_individuals=20 threaded n_each=nothing > experiments/logs/grid_search_linearly_separable_threaded.log
