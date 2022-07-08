#!/bin/bash
#SBATCH --begin=now
#SBATCH --job-name=rgb-footprint-extract
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nathanjo@law.stanford.edu
#SBATCH --partition=owners
#SBATCH --time=0:30:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

cd ../../

singularity exec --nv $GROUP_HOME/singularity/rgb-building1.sif python3 run_deeplab.py --inference --backbone=drn_c42 --out-stride=8 \
    --workers=2 --epochs=1 --test-batch-size=1 --gpu-ids=0 --resume=LA_OSM_0.1_True_0.001_0.0001_crowdAI --best-miou \
    --window-size=256 --stride=256 \
    --input-filename='sj_naip_subset.npy' \
    --output-filename='LA_infer_sj_naip_subset.png'

