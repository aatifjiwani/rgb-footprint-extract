#!/bin/bash
#SBATCH --begin=now
#SBATCH --job-name=rgb-footprint-extract
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nathanjo@law.stanford.edu
#SBATCH --partition=owners
#SBATCH --time=03:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1

cd ../../

singularity exec --nv $GROUP_HOME/singularity/rgb-building1.sif python3 run_deeplab.py --minference --backbone=drn_c42 --out-stride=8 \
    --workers=2 --epochs=1 --test-batch-size=1 --gpu-ids=0 --resume=SJ_0.2_True_0.0005_0.0001_1.03_24_superresx2 \
    --window-size=256 --stride=256 --best-miou --dataset=OSM \
    --data-root='/oak/stanford/groups/deho/building_compliance/san_jose_naip_512/phase2_superresx2/' \
    --output-dir='/oak/stanford/groups/deho/building_compliance/san_jose_naip_512/phase2_superresx2/infer'
