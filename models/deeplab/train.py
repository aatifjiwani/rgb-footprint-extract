# We would like to thank and acknowledge jfzhang95 for the DeepLabV3+ module
# as well as a template for metrics and the training pipeline.
# His code repository can be found here:
# https://github.com/jfzhang95/pytorch-deeplab-xception

import numpy as np

import torch
import pytorch_lightning as pl

from models.deeplab.modeling.deeplab import DeepLab
from models.utils.loss import CELoss, DICELoss, CE_DICELoss, MSELoss
from models.utils.metrics import Evaluator
# from models.utils.scheduler import LR_Scheduler


class DeepLabModule(pl.LightningModule):
    def __init__(self,
                 args,
                 ):
        super().__init__()
        self.save_hyperparameters()

        # setup params
        self.dataset = args.dataset
        self.incl_bounds = args.incl_bounds
        self.nclass = args.num_classes
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.nesterov = args.nesterov
        self.lr = args.lr
        self.lr_scheduler = args.lr_scheduler
        self.epochs = args.epochs
        self.steps_per_epoch = args.steps_per_epoch

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

        # setup criterion
        if self.incl_bounds:
            assert args.loss_type in ["wce_dice"]

        if args.loss_type == "ce":
            self.criterion = CELoss(weight=args.loss_weights)
        elif args.loss_type == "dice":
            self.criterion = DICELoss(weight=args.loss_weights)
        elif args.loss_type == "ce_dice":
            self.criterion = CE_DICELoss(weight=args.loss_weights)
        elif args.loss_type == "mse":
            self.criterion = MSELoss(weight=args.loss_weights)
        else:
            raise ValueError("loss_type %s is not supported" % args.loss_type)

        self.evaluator = Evaluator(self.nclass)

    def configure_optimizers(self):
        # train_params = [
        #     {'params': self.model.get_1x_lr_params(), 'lr': self.lr},
        #     {'params': self.model.get_10x_lr_params(), 'lr': self.lr * 10}
        # ]

        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=self.lr,
                                    momentum=self.momentum,
                                    weight_decay=self.weight_decay,
                                    nesterov=self.nesterov)

        # scheduler = LR_Scheduler(self.lr_scheduler, 
        #                          self.lr, 
        #                          self.epochs, 
        #                          self.steps_per_epoch)

        # return [optimizer], [scheduler]
        return optimizer

    def forward(self, x):
        output = self.model(x)
        return output
    
    def training_step(self, batch, batch_idx):
        image, mask = batch['image'], batch['mask'].long()
        if self.incl_bounds:
            boundary_weights = batch['boundary']
        else:
            boundary_weights = None

        # need to squeeze if combined dataset
        if self.dataset == "combined":
            image, mask = image.squeeze(), mask.squeeze()

        output = self(image)

        if boundary_weights is not None:
            loss = self.criterion(output, mask, boundary_weights)
        else:
            loss = self.criterion(output, mask)

        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        image, mask = batch['image'], batch['mask'].long()
        if self.incl_bounds:
            boundary_weights = batch['boundary']
        else:
            boundary_weights = None

        # need to squeeze if combined dataset
        if self.dataset == "combined":
            image, mask = image.squeeze(), mask.squeeze()

        output = self(image)
        if boundary_weights is not None:
            loss = self.criterion(output, mask, boundary_weights)
        else:
            loss = self.criterion(output, mask)

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

