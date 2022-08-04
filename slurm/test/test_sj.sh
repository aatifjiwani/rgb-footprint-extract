#!/bin/bash
#SBATCH --begin=now
#SBATCH --job-name=rgb-footprint-extract
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nathanjo@law.stanford.edu
#SBATCH --partition=owners
#SBATCH --time=00:30:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1

cd ../../

singularity exec --nv $GROUP_HOME/singularity/rgb-building1.sif python3 run_deeplab.py --evaluate --backbone=drn_c42 --out-stride=8 --dataset=OSM \
    --workers=4 --epochs=1 --test-batch-size=4 --gpu-ids=0 --resume=OSM_trf --best-miou \
    --checkname=eval_OSM/default_no_freezebn --data-root=/oak/stanford/groups/deho/building_compliance/san_jose_naip_512/
