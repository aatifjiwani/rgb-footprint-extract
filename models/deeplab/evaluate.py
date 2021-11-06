# https://github.com/jfzhang95/pytorch-deeplab-xception
import os
import time
import wandb
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader

from models.deeplab.modeling.sync_batchnorm.replicate import patch_replication_callback
from models.deeplab.modeling.deeplab import *
from models.utils.loader import load_model
from models.utils.loss import SegmentationLosses
from models.utils.saver import Saver
from models.utils.metrics import Evaluator
from models.utils.collate_fn import generate_split_collate_fn, handle_concatenation
from models.utils.custom_transforms import tensor_resize

from datasets import build_test_dataloader


class Tester(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        
        # Define transforms and Dataloader
        deeplab_collate_fn = None
        transform = None
        test_dataset = build_test_dataloader(args.dataset, args.data_root, transform)

        print("Testing on {} samples".format(len(test_dataset)))
        self.test_loader = DataLoader(
                                    test_dataset, 
                                    batch_size=args.test_batch_size, 
                                    shuffle=True, 
                                    num_workers=args.workers,
                                    collate_fn=deeplab_collate_fn
                                )
        self.nclass = args.num_classes

        # Define network
        print("Using backbone {} with output stride {} and dropout values {}, {}".format(args.backbone, args.out_stride, args.dropout[0], args.dropout[1]))
        self.model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn,
                        dropout_low=args.dropout[0],
                        dropout_high=args.dropout[1],
                    )
                    
        self.model = load_model(self.model, args.resume, args.best_miou, args.cuda, args.gpu_ids)

        self.model.eval()
        self.evaluator = Evaluator(self.nclass)
        self.curr_step = 0

    def test(self, ):
        tbar = tqdm(self.test_loader)

        total_pixelAcc = []
        total_mIOU = []
        total_dice = []
        total_f1 = []
        total_precision, total_recall = [], []

        for i, sample in enumerate(tbar):
            image, mask = sample['image'], sample['mask'].long()
            names = sample['name']

            # cuda enable image/mask
            if self.args.cuda:
                image, mask = image.cuda(), mask.cuda()

            with torch.no_grad():
                output = self.model(image)
                     
            pred = torch.nn.functional.softmax(output, dim=1)
            pred = pred.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)

            target = mask.cpu().numpy()

            total_pixelAcc.append(self.evaluator.pixelAcc_manual(target, pred))
            total_mIOU.append(self.evaluator.mIOU_manual(target, pred))
            f1, pre, rec = self.evaluator.f1score_manual_full(target, pred)

            if (not np.isnan(f1) and not np.isnan(pre) and not np.isnan(rec)):
                total_f1.append(f1)
                total_precision.append(pre)
                total_recall.append(rec)
            else:
                total_f1.append(0)
                total_precision.append(0)
                total_recall.append(0)

        print({
                "test_mIOU": np.mean(total_mIOU),
                "test_pixel_acc": np.mean(total_pixelAcc),
                "test_f1": np.mean(total_f1),
                "test_ap": np.mean(total_precision),
                "test_ar": np.mean(total_recall),
            })





