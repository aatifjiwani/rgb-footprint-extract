python run_deeplab.py^
 --inference^
 --model-path=weights\_caugiay_deeplab_resnet50_ce_dice_f0.5_sz1024\best_loss-epoch=14-train_loss=0.10-val_loss=0.17.ckpt^
 --output-dir=run\_caugiay_deeplab_resnet50_ce_dice_f0.5_sz1024\best_loss-epoch=14-train_loss=0.10-val_loss=0.17\conf=0.0\cauGiay\^
 --workers=8^
 --dataset=cauGiay^
 --test-batch-size=16^
 --gpu-ids=0^
 --conf-t=0.0^
 --data-root=data\
