# Ensure all pretrained weights are located in the weights/ directory

# Example for sample Urban3D image
CUDA_VISIBLE_DEVICES=0 python3 run_deeplab.py --inference --backbone=drn_c42 --out-stride=8 \
    --workers=2 --epochs=1 --test-batch-size=1 --gpu-ids=0 --resume=urban3d --best-miou \
    --window-size=512 --stride=512 \
    --input-filename='/data/Urban3D/test/images/JAX_Tile_027.npy' \
    --output-filename='output.png' 
