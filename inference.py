# We would like to thank and acknowledge jfzhang95 for the DeepLabV3+ module 
# as well as a template for metrics and the training pipeline. 
# His code repository can be found here:
# https://github.com/jfzhang95/pytorch-deeplab-xception

from models.deeplab.train import *
from models.deeplab.evaluate import *
import argparse
import os
import time
import rasterio

def main():
    parser = argparse.ArgumentParser(description="Building footprint segmentation inference script")

    parser.add_argument('--backbone', type=str, default='drn_c42', choices=['resnet', 'xception', 'drn', 'mobilenet', 'drn_c42'], help='backbone name (default: drn_c42)')
    parser.add_argument('--out-stride', type=int, default=8, help='network output stride (default: 8)')
    parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
    parser.add_argument('--sync-bn', type=bool, default=None, help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False, help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce_dice', choices=['ce', 'ce_dice', 'wce_dice'], help='loss func type (default: ce)')
    parser.add_argument("--num-classes", type=int, default=2, help='number of classes to predict (2 for binary mask)')
    parser.add_argument('--dropout', type=float, nargs="+", default=[0.1, 0.5], help='dropout values')

    # training hyper params
    parser.add_argument('--test-batch-size', type=int, default=8, metavar='N', help='input batch size for testing (default: auto)')

    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0', help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

    # name
    parser.add_argument('--checkname', type=str, default="_evaluation_crowdAI", help='set the checkpoint name')
    parser.add_argument('--checkpoint-root', type=str, default='weights', help='the root directory of the checkpoints')

    parser.add_argument('--resume', type=str, default="crowdAI", choices=["crowdAI", "spaceNet", "urban3d"], help='experiment to load')
    parser.add_argument('--best-miou', action='store_true', default=False)

    #boundaries
    parser.add_argument('--incl-bounds', action='store_true', default=False, help='includes boundaries of masks in loss function')
    parser.add_argument('--bounds-kernel-size', type=int, default=3, help='kernel size for calculating boundary')

    parser.add_argument('--input-fn', type=str, required=True, help="path to an input GeoTIFF to run model on")
    parser.add_argument('--output-fn', type=str, required=True, help="path to write predictions as a GeoTIFF")
    parser.add_argument('--chip-size', type=int, default=1024, help="the size of input patches to sample from the input")
    parser.add_argument('--padding', type=int, default=256, help="the number of pixels on each edge to downweight predictions to help prevent checkerboarding")
    parser.add_argument('--switch-r-and-b', action='store_true', help='whether to switch the first and third bands of the input (for BGR formatted TIFFs)')


    args = parser.parse_args()

    args.dataset = "tile"
    args.use_wandb = False

    # Parse inputs
    assert os.path.exists(args.input_fn)
    assert not os.path.exists(args.output_fn)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    args.chip_stride = args.chip_size - 2 * args.padding
    assert args.chip_size > 0

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)

    # Run inference
    torch.manual_seed(args.seed)
    tester = Tester(args)

    with rasterio.open(args.input_fn) as f:
        input_width, input_height = f.width, f.height
        input_profile = f.profile.copy()

    output_hard = tester.inference(input_width, input_height)

    input_profile["count"] = 1
    input_profile["tiled"] = True
    input_profile["compression"] = "lzw"
    input_profile["predictor"] = 2

    with rasterio.open(args.output_fn, "w", **input_profile) as f:
        f.write(output_hard, 1)
        f.write_colormap(1, {
            0: (0, 0, 0, 0),
            1: (255, 0, 0, 255)
        })


if __name__ == "__main__":
    main()
