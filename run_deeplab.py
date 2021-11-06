# We would like to thank and acknowledge jfzhang95 for the DeepLabV3+ module 
# as well as a template for metrics and the training pipeline. 
# His code repository can be found here:
# https://github.com/jfzhang95/pytorch-deeplab-xception

from models.deeplab.train import *
from models.deeplab.evaluate import *
import argparse

def main():
    parser = argparse.ArgumentParser(description="DeeplabV3+ And Evaluation")

    # model parameters

    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet', 'drn_c42'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='urban3d',
                        choices=['urban3d', 'spaceNet', 'crowdAI', 'combined'],
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
                        choices=['ce', 'ce_dice', 'wce_dice'],
                        help='loss func type (default: ce)')
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
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # name
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')

    # evaluation option
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    parser.add_argument('--use-wandb', action='store_true', default=False)

    parser.add_argument('--resume', type=str, default=None, help='experiment to load')
    parser.add_argument("--evaluate", action='store_true', default=False)
    parser.add_argument('--best-miou', action='store_true', default=False)

    #boundaries
    parser.add_argument('--incl-bounds', action='store_true', default=False,
                        help='includes boundaries of masks in loss function')
    parser.add_argument('--bounds-kernel-size', type=int, default=3,
                        help='kernel size for calculating boundary')

    args = parser.parse_args()
    run_deeplab(args)

def run_deeplab(args):
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

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        raise ValueError("epochs must be specified")

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)

    torch.manual_seed(args.seed)
    if args.evaluate:
        handle_evaluate(args)
    else:
        handle_training(args)

def handle_evaluate(args):
    tester = Tester(args)
    print("Experiment {} instantiated. Training starting...".format(args.checkname))

    tester.test()

def handle_training(args):
    trainer = Trainer(args)

    print("Learning rate: {}; L2 factor: {}".format(args.lr, args.weight_decay))
    print("Experiment {} instantiated. Training starting...".format(args.checkname))
    print("Training for {} epochs".format(trainer.args.epochs))
    print("Batch size: {}; Test Batch Size: {}".format(args.batch_size, args.test_batch_size))
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val:
            trainer.validation(epoch)

if __name__ == "__main__":
    main()
