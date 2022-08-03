#!/bin/bash
#SBATCH --begin=now
#SBATCH --job-name=rgb-footprint-extract
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nathanjo@law.stanford.edu
#SBATCH --partition=owners
#SBATCH --time=00:05:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

cd ../../

singularity exec --nv $GROUP_HOME/singularity/rgb-building1.sif python3 run_deeplab.py --inference --backbone=drn_c42 --out-stride=8 \
    --workers=2 --epochs=1 --test-batch-size=1 --gpu-ids=0 --resume=SJ_0.1_True_0.0005_0.0001_1.0_20_superresx4 \
    --window-size=256 --stride=256 --best-miou \
    --input-filename='/oak/stanford/groups/deho/building_compliance/san_jose_naip_512/phase2_superresx4/temp/3.npy' \
    --output-filename='p2_sj_adu_superresx4_3.npy'