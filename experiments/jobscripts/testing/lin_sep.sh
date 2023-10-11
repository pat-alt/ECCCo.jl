#!/bin/bash

#SBATCH --job-name="Grid-search Linearly Separable (ECCCo)"
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=4
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=4GB
#SBATCH --account=research-eemcs-insy
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 

module load 2023r1 julia
module load openmpi

set -x
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

srun julia --project=experiments --threads $SLURM_CPUS_PER_TASK experiments/run_experiments.jl -- data=linearly_separable output_path=results mpi grid_search n_individuals=10 threaded > experiments/grid_search_linearly_separable.log
