# Ensure all pretrained weights are located in the weights/ directory

# Example for CrowdAI
CUDA_VISIBLE_DEVICES=0 python3 run_deeplab.py --evaluate --backbone=drn_c42 --out-stride=8 \
    --workers=2 --epochs=1 --test-batch-size=3 --gpu-ids=0 \
    --checkname=_evaluation_crowdAI --dataset=crowdAI --resume=crowdAI 
