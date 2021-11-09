# We would like to thank and acknowledge jfzhang95 for the DeepLabV3+ module 
# as well as a template for metrics and the training pipeline. 
# His code repository can be found here:
# https://github.com/jfzhang95/pytorch-deeplab-xception

import os

import torch
import pytorch_lightning as pl

from models.deeplab.train import DeepLabModule
from datasets import DeepLabDataModule

import argparse


def _parse_args():
    parser = argparse.ArgumentParser(description="DeeplabV3+ And Evaluation")

    # model parameters

    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet', 'drn_c42'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=8,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='urban3d',
                        choices=['urban3d', 'spaceNet', 'crowdAI', 'cauGiay', 'combined'],
                        help='dataset name (default: urban3d)')
    parser.add_argument('--data-root', type=str, default='/data/',
                        help='datasets root path')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce_dice',
                        choices=['ce', 'dice', 'ce_dice', 'wce_dice'],
                        help='loss func type (default: ce_dice)')
    parser.add_argument('--fbeta', type=float, default=1, help='beta for FBeta-Measure')
    parser.add_argument('--loss-weights', type=float, nargs="+", default=[1.0, 1.0], 
                        help='loss weighting')
    parser.add_argument("--num-classes", type=int, default=2, 
                        help='number of classes to predict (2 for binary mask)')
    parser.add_argument('--dropout', type=float, nargs="+", default=[0.1, 0.5], 
                    help='dropout values')

    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: auto)')
    # optimizer params
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')

    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', default=0, type=int, nargs="+",
                        help='use which gpu to train (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # name
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    parser.add_argument('--resume', type=str, default=None, help='resume training')

    # evaluation option
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    parser.add_argument('--best-miou', action='store_true', default=False)

    # inference options
    parser.add_argument("--inference", action='store_true', default=False)
    parser.add_argument('--model-path', type=str, default=None, help='trained model for inference')
    parser.add_argument('--output-dir', type=str, default=".", help='store predictions')

    #boundaries
    parser.add_argument('--incl-bounds', action='store_true', default=False,
                        help='includes boundaries of masks in loss function')
    parser.add_argument('--bounds-kernel-size', type=int, default=3,
                        help='kernel size for calculating boundary')

    args = parser.parse_args()
    return args


def main():
    args = _parse_args()
    run_deeplab(args)


def run_deeplab(args):
    if args.inference:
        handle_inference(args)
    else:
        handle_training(args)


def handle_inference(args):
    assert args.model_path is not None

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    dm = DeepLabDataModule(args)
    dm.setup("inference")

    model = DeepLabModule.load_from_checkpoint(args.model_path)

    from models.deeplab.inference import SemanticSegmentationTask
    task = SemanticSegmentationTask(model, args.output_dir)

    trainer = pl.Trainer(gpus=args.gpu_ids)
    trainer.predict(task, datamodule=dm)
    

def handle_training(args):
    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        raise ValueError("epochs must be specified")

    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)

    print("Learning rate: {}; L2 factor: {}".format(args.lr, args.weight_decay))
    print("Experiment {} instantiated. Training starting...".format(args.checkname))
    print("Training for {} epochs".format(args.epochs))
    print("Batch size: {}; Test Batch Size: {}".format(args.batch_size, args.test_batch_size))
    torch.manual_seed(args.seed)
    
    dm = DeepLabDataModule(args)
    dm.setup("fit")

    model = DeepLabModule(args)
    
    # from torchsummary import summary
    # summary(model, (3, 650, 650), device="cpu")

    # define callbacks
    from pytorch_lightning.callbacks import ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join('weights', args.checkname),
        filename="best_loss-{epoch:02d}-{train_loss:.2f}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
        save_on_train_epoch_end=True
    )

    # from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    # early_stop_callback = EarlyStopping(monitor="val_loss", 
    #                                     min_delta=0.01, 
    #                                     patience=3, 
    #                                     verbose=False, 
    #                                     mode="min")

    callbacks=[checkpoint_callback]

    # use float (e.g 1.0) to set val frequency in epoch
    # if val_check_interval is integer, val frequency is in batch step
    training_params = {
        "callbacks": callbacks,
        "gpus": args.gpu_ids,
        "val_check_interval": 1.0,
        "max_epochs": args.epochs
    }
    if args.resume is not None:
        ckpt_path = os.path.join('weights', args.checkname, args.resume)
        training_params["resume_from_checkpoint"] = ckpt_path
    
    trainer = pl.Trainer(**training_params)
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
