# https://github.com/jfzhang95/pytorch-deeplab-xception
import os
import time
import wandb
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader

from models.deeplab.modeling.sync_batchnorm.replicate import patch_replication_callback
from models.deeplab.modeling.deeplab import *
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
        if args.switch_r_and_b:
            transform = lambda x: x[[2,1,0],:,:].copy()
        else:
            transform = lambda x: x[:3,:,:].copy()
        test_dataset = build_test_dataloader(args, transform)

        print("Testing on {} samples".format(len(test_dataset)))
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.workers,
        )
        self.nclass = args.num_classes

        # Define network
        print("Using backbone {} with output stride {} and dropout values {}, {}".format(args.backbone, args.out_stride, args.dropout[0], args.dropout[1]))
        self.model = DeepLab(
            num_classes=self.nclass,
            backbone=args.backbone,
            output_stride=args.out_stride,
            sync_bn=args.sync_bn,
            freeze_bn=args.freeze_bn,
            dropout_low=args.dropout[0],
            dropout_high=args.dropout[1],
        )

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resume
        checkpoint_name = "best_loss_checkpoint.pth.tar"
        if args.best_miou:
            checkpoint_name = "best_miou_checkpoint.pth.tar"

        checkpoint_path = os.path.join(args.checkpoint_root, args.resume, checkpoint_name)
        print("Resuming from {}".format(checkpoint_path))

        model_checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(model_checkpoint)

        self.model.eval()
        self.evaluator = Evaluator(self.nclass)
        self.curr_step = 0

    def test(self):
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

    def inference(self, input_width, input_height):
        assert self.test_loader.dataset.__class__.__name__ == "TileInferenceDataset"

        tbar = tqdm(self.test_loader)

        output = np.zeros((2, input_height, input_width), dtype=np.float32)
        kernel = np.ones((self.args.chip_size, self.args.chip_size), dtype=np.float32)
        kernel[self.args.padding:-self.args.padding, self.args.padding:-self.args.padding] = 5
        counts = np.zeros((input_height, input_width), dtype=np.float32)

        for i, (data, coords) in enumerate(tbar):
            if self.args.cuda:
                data = data.cuda()

            with torch.no_grad():
                t_output = self.model(data)
                t_output = torch.nn.functional.softmax(t_output, dim=1).cpu().numpy()

            for j in range(t_output.shape[0]):
                y, x =  coords[j]

                output[:, y:y+self.args.chip_size, x:x+self.args.chip_size] += t_output[j] * kernel
                counts[y:y+self.args.chip_size, x:x+self.args.chip_size] += kernel

        output = output / counts
        output_hard = output.argmax(axis=0).astype(np.uint8)
        return output_hard

