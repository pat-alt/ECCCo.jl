#!/bin/bash

#SBATCH --job-name="Grid-search Linearly Separable (ECCCo)"
#SBATCH --time=01:00:00
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --mem-per-cpu=4GB
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 

module use /opt/insy/modulefiles          # Use DAIC INSY software collection
module load openmpi

export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

srun julia --project=experiments experiments/run_experiments.jl -- data=linearly_separable output_path=results mpi grid_search n_individuals=10 threaded > experiments/grid_search_linearly_separable.log
