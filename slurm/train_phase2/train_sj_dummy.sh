#!/bin/bash
#SBATCH --begin=now
#SBATCH --job-name=rgb-footprint-extract
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nathanjo@law.stanford.edu
#SBATCH --partition=owners
#SBATCH --time=00:30:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

cd ../../

singularity exec --nv $GROUP_HOME/singularity/rgb-building1.sif python3 run_deeplab.py --backbone=drn_c42 --out-stride=8 --dataset=OSM \
    --workers=4 --loss-type=wce_dice --fbeta=0.1 --epochs=1 --batch-size=2 --test-batch-size=4 --weight-decay=1e-4 \
    --gpu-ids=0 --lr=1e-4 --loss-weights 1.0 1.0 --dropout 0.3 0.5 --incl-bounds --resume=crowdAI --best-miou --freeze-bn \
    --data-root=/oak/stanford/groups/deho/building_compliance/san_jose_naip_512/phase2_dummy/ --loss-weights-param=1.01 