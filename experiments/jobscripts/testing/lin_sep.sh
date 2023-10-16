#!/bin/bash

#SBATCH --job-name="Grid-search Linearly Separable (ECCCo)"
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=4GB
#SBATCH --account=research-eemcs-insy
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 

module load 2023r1 openmpi

source experiments/slurm_header.sh

srun julia --project=experiments experiments/run_experiments.jl -- data=linearly_separable output_path=results mpi grid_search n_individuals=1 > experiments/grid_search_linearly_separable.log
