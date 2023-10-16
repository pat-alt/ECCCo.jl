#!/bin/bash

#SBATCH --job-name="Grid-search Moons (ECCCo)"
#SBATCH --time=01:00:00
#SBATCH --ntasks=30
#SBATCH --cpus-per-task=30
#SBATCH --partition=general
#SBATCH --mem-per-cpu=1GB
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 

module use /opt/insy/modulefiles          # Use DAIC INSY software collection
module load openmpi

source experiments/slurm_header.sh

srun julia --project=experiments --threads $SLURM_CPUS_PER_TASK experiments/run_experiments.jl -- data=moons output_path=results mpi grid_search threaded n_individuals=100 > experiments/logs/grid_search_moons.log