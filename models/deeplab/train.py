# We would like to thank and acknowledge jfzhang95 for the DeepLabV3+ module
# as well as a template for metrics and the training pipeline.
# His code repository can be found here:
# https://github.com/jfzhang95/pytorch-deeplab-xception

import os
import numpy as np

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from models.deeplab.modeling.deeplab import *
from models.utils.loss import SegmentationLosses
from models.utils.metrics import Evaluator

from datasets import build_dataloader


class DeepLabModule(pl.LightningModule):
    def __init__(self,
                 args):
        self.save_hyperparameters()

        # setup params
        self.dataset = args.dataset
        self.incl_bounds = args.incl_bounds
        self.nclass = args.num_classes
        self.momentum=args.momentum
        self.weight_decay=args.weight_decay
        self.nesterov=args.nesterov
        self.lr = args.lr

        # setup model
        print("Using backbone {} with output stride {} and dropout values {}, {}"
                .format(args.backbone, args.out_stride, args.dropout[0], args.dropout[1]))
        
        self.model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn,
                        dropout_low=args.dropout[0],
                        dropout_high=args.dropout[1],
                    )

        if args.pretrained is not None:
            print("Loading pretrained model from {}".format(args.pretrained))
            model_checkpoint = torch.load(args.pretrained)
            self.model.load_state_dict(model_checkpoint)

        # setup criterion
        if self.incl_bounds:
            assert args.loss_type in ["wce_dice"]

        self.criterion = SegmentationLosses(beta=args.fbeta,
                                            weight=args.loss_weights,
                                            cuda=args.cuda).build_loss(mode=args.loss_type)

        self.evaluator = Evaluator(self.nclass)

    def configure_optimizers(self):
        train_params = [
            {'params': self.model.get_1x_lr_params(), 'lr': self.lr},
            {'params': self.model.get_10x_lr_params(), 'lr': self.lr * 10}
        ]

        optimizer = torch.optim.SGD(train_params,
                                    momentum=self.momentum,
                                    weight_decay=self.weight_decay,
                                    nesterov=self.nesterov)

        return optimizer

    def forward(self, x):
        output = self.model(x)
        return output

    def _step(self, image, mask):
        output = self(image)

        if self.incl_bounds:
            boundary_weights = image['boundary']
            loss = self.criterion(output, mask, boundary_weights)
        else:
            loss = self.criterion(output, mask)

        return output, loss

    def training_step(self, batch, batch_idx):
        image, mask = batch['image'], batch['mask'].long()
        # need to squeeze if combined dataset
        if self.dataset == "combined":
            image, mask = image.squeeze(), mask.squeeze()

        _, loss = self._step(image, mask)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        image, mask = batch['image'], batch['mask'].long()
        # need to squeeze if combined dataset
        if self.dataset == "combined":
            image, mask = image.squeeze(), mask.squeeze()

        output, loss = self._step(image, mask)

        pred = torch.nn.functional.softmax(output, dim=1)
        pred = pred.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)

        target = mask.cpu().numpy()

        pixel_acc = self.evaluator.pixelAcc_manual(target, pred)
        val_miou = self.evaluator.mIOU_manual(target, pred)
        f1_score = self.evaluator.f1score_manual(target, pred)

        return loss, pixel_acc, val_miou, f1_score

    def validation_epoch_end(self, outs):
        total_loss = []
        total_pixelAcc = []
        total_mIOU = []
        total_f1 = []
        for (loss, pixel_acc, val_miou, f1) in outs:
            total_loss.append(loss.item())
            total_pixelAcc.append(pixel_acc)
            total_mIOU.append(val_miou)
            if f1 is not None:
                total_f1.append(f1)
            else:
                total_f1.append(0)

        self.log('val_loss', np.mean(total_loss))
        self.log('val_pixel_acc', np.mean(total_mIOU))
        self.log('val_miou', np.mean(total_pixelAcc))
        self.log('val_f1_score', np.mean(total_f1))


class DeepLabDataModule(pl.LightningDataModule):
    def __init__(self,
                 args
                 ):
        super().__init__()

        self.dataset = args.dataset
        self.data_root = args.data_root
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.workers = args.workers

        # Define transforms and Dataloader
        self.boundary_ks = args.bounds_kernel_size if args.incl_bounds else None
        self.deeplab_collate_fn = None
        self.transform = None

    def prepare_data(self):
        # called only on 1 GPU
        pass

    def setup(self,
              stage: str = "fit"):
        resize = 2048

        if stage == "fit" or stage is None:
            split = 2
            self.train_dataset, self.val_dataset = build_dataloader(self.dataset,
                                                                    self.data_root,
                                                                    self.boundary_ks,
                                                                    self.transform,
                                                                    resize,
                                                                    split)

    def train_dataloader(self):
        return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.workers,
                collate_fn=self.deeplab_collate_fn
            )

    def val_dataloader(self):
        return DataLoader(
                self.val_dataset,
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=self.workers,
                collate_fn=self.deeplab_collate_fn
            )