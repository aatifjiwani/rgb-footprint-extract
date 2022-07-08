#!/bin/bash
#SBATCH --begin=now
#SBATCH --job-name=rgb-footprint-extract
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nathanjo@law.stanford.edu
#SBATCH --partition=owners
#SBATCH --time=0:30:00
#SBATCH --mem=16GB

cd ../../

singularity exec $GROUP_HOME/singularity/rgb-building1.sif python3 run_deeplab.py --inference --backbone=drn_c42 --out-stride=8 \
    --workers=2 --epochs=1 --test-batch-size=1 --gpu-ids=0 --resume=urban3d --best-miou \
    --window-size=512 --stride=512 \
    --input-filename='data/Urban3D/test/images/JAX_Tile_000.npy' \
    --output-filename='output_urban3d.png' 

