# Example for Urban3D
CUDA_VISIBLE_DEVICES=0 python3 run_deeplab.py --backbone=drn_c42 --out-stride=8 --dataset=urban3d \
    --workers=4 --loss-type=wce_dice --fbeta=0.5 --epochs=60 --batch-size=4 --test-batch-size=4 --weight-decay=1e-4 \
    --gpu-ids=0 --lr=1e-3 --loss-weights 1.0 1.0 --dropout 0.3 0.5 --incl-bounds \
    --checkname=_urban3d_deeplab_drn_c42_wce_dicef0.5 --data-root=/data/