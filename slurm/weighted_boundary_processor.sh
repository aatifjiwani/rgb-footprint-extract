#!/bin/bash
#SBATCH --begin=now
#SBATCH --job-name=rgb-footprint-extract
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nathanjo@law.stanford.edu
#SBATCH --partition=owners
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB

cd ../

singularity exec $GROUP_HOME/singularity/rgb-building1.sif python3 datasets/converters/weighted_boundary_processor_parallel.py

