#!/bin/bash
#SBATCH --begin=now
#SBATCH --job-name=rgb-footprint-extract
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nathanjo@law.stanford.edu
#SBATCH --partition=owners
#SBATCH --time=12:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:3

cd ../../

singularity exec --nv $GROUP_HOME/singularity/rgb-building1.sif python3 run_deeplab.py --backbone=drn_c42 --out-stride=8 --dataset=OSM_split4 \
    --workers=4 --loss-type=wce_dice --fbeta=0.1 --epochs=60 --batch-size=24 --test-batch-size=4 --weight-decay=1e-4 \
    --gpu-ids=0,1,2 --lr=5e-4 --loss-weights 1.0 1.0 --dropout 0.3 0.5 --incl-bounds --resume=crowdAI --best-miou --freeze-bn \
    --data-root=/oak/stanford/groups/deho/building_compliance/san_jose_naip_512/phase2_superresx4/ --loss-weights-param=1 --preempt-robust \
    --superres 4 --use-wandb --checkname-add=regenerated_wt