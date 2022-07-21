# We would like to thank and acknowledge jfzhang95 for the DeepLabV3+ module 
# as well as a template for metrics and the training pipeline. 
# His code repository can be found here:
# https://github.com/jfzhang95/pytorch-deeplab-xception

import os
import time
import wandb
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader

from models.deeplab.modeling.sync_batchnorm.replicate import patch_replication_callback
from models.deeplab.modeling.deeplab import *
from models.utils.loss_p2 import SegmentationLosses
from models.utils.loader import load_model
from models.utils.saver import Saver
from models.utils.metrics import Evaluator
from models.utils.collate_fn import generate_split_collate_fn, handle_concatenation
from models.utils.custom_transforms import tensor_resize
from models.utils.wandb_utils import *

from datasets import build_dataloader
import json

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        
        # Define transforms and Dataloader
        boundary_ks = args.bounds_kernel_size if args.incl_bounds else None
        deeplab_collate_fn = None
        transform = None
        
        resize = 2048
        split = 2
        train_dataset, val_dataset = build_dataloader(args.dataset, args.data_root, boundary_ks, transform, resize, split)

        print("Training on {} samples, Validating on {} samples".format(len(train_dataset), len(val_dataset)))
        self.validation_loader = DataLoader(
                                    val_dataset, 
                                    batch_size=args.test_batch_size, 
                                    shuffle=True, 
                                    num_workers=args.workers,
                                    collate_fn=deeplab_collate_fn
                                )
        self.train_loader = DataLoader(
                                train_dataset, 
                                batch_size=args.batch_size, 
                                shuffle=True, 
                                num_workers=args.workers,
                                collate_fn=deeplab_collate_fn    
                            )

        # self.validation_loader = DataLoader(
        #                             val_dataset, 
        #                             batch_size=args.test_batch_size, 
        #                             num_workers=args.workers,
        #                             collate_fn=deeplab_collate_fn
        #                         )
        # self.train_loader = DataLoader(
        #                         train_dataset, 
        #                         batch_size=args.batch_size, 
        #                         num_workers=args.workers,
        #                         collate_fn=deeplab_collate_fn    
        #                     )


        self.nclass = args.num_classes

        # Define network
        print("Using backbone {} with output stride {} and dropout values {}, {}".format(args.backbone, args.out_stride, args.dropout[0], args.dropout[1]))
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn,
                        dropout_low=args.dropout[0],
                        dropout_high=args.dropout[1],
                    )

        # train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
        #                 {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        # train_params = [{'params': model.get_final_lr_params(), 'lr': args.lr},
        #                 {'params': model.get_0x_lr_params(), 'lr': 0}]

        # Define Optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

        if args.incl_bounds:
            assert args.loss_type in ["wce_dice"]

        # freeze all but last layer
        for name, param in model.backbone.named_parameters():
            param.requires_grad = False

        for name, param in model.aspp.named_parameters():
            param.requires_grad = False

        for name, module in model.decoder.named_modules():
            if 'last_conv' not in name and name != '':
                for p in module.parameters():
                    p.requires_grad = False
            elif 'last_conv' in name:
                if args.freeze_bn:
                    if isinstance(module, SynchronizedBatchNorm2d) \
                            or isinstance(module, nn.BatchNorm2d):
                        for p in module.parameters():
                            p.requires_grad = False

        self.criterion = SegmentationLosses(beta=args.fbeta, weight=args.loss_weights, cuda=args.cuda,
            loss_weights_param=args.loss_weights_param).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer

        if args.use_wandb:
            wandb.watch(self.model)

        self.model, self.optimizer, self.start_epoch = load_model(self.model, self.optimizer, args.resume, args.best_miou, args.cuda, args.gpu_ids)

        # if we are resuming training
        if args.preempt_robust and self.start_epoch > 0:
            # check if jsonl exists -- it should, but adding this if statement as a precaution
            jsonl_fp = os.path.join('/oak/stanford/groups/deho/building_compliance/rgb-footprint-extract/run', args.checkname, 'metrics.jsonl')
            if os.path.exists(jsonl_fp):
                # open jsonl 
                val_metrics_historical = {'loss': [], 'mIOU': [], 'pixel_acc': [], 'f1': []}
                train_metrics_historical = {}
                times = []
                with open(jsonl_fp, 'r') as f:
                    for line in f:
                        l = json.loads(line)
                        if 'val_loss' in l:
                            val_metrics_historical['loss'].append(l['val_loss'])
                            val_metrics_historical['mIOU'].append(l['val_mIOU'])
                            val_metrics_historical['pixel_acc'].append(l['val_pixel_acc'])
                            val_metrics_historical['f1'].append(l['val_f1'])
                        elif 'train_loss' in l:
                            if l['epoch'] not in train_metrics_historical:
                                train_metrics_historical[l['epoch']] = {'loss': [l['train_loss']], 'mIOU': [l['mIOU']], 
                                                                        'pixel_acc': [l['pixel_acc']], 'f1': [l['f1']]}
                            else:
                                train_metrics_historical[l['epoch']]['loss'].append(l['train_loss'])
                                train_metrics_historical[l['epoch']]['mIOU'].append(l['mIOU'])
                                train_metrics_historical[l['epoch']]['pixel_acc'].append(l['pixel_acc'])
                                train_metrics_historical[l['epoch']]['f1'].append(l['f1'])
                        else:
                            times.append(l)

                # assert that length of validation set is the same as start_epoch
                assert len(val_metrics_historical['loss']) == self.start_epoch

                # EDGE CASE: if pre-empted mid-way through, truncate that from the jsonl file.
                if max(train_metrics_historical.keys()) == self.start_epoch: # this means that we have to drop the last key
                    train_metrics_historical.pop(max(train_metrics_historical.keys()), None)

                    # rewrite jsonl file
                    with open(jsonl_fp, 'w') as f:
                        e = 0
                        while e < len(val_metrics_historical['loss']): # need to think about this condition a little more
                            for idx, element in enumerate(train_metrics_historical[e]['loss']):
                                dic_line = {'train_loss': element, 'mIOU': train_metrics_historical[e]['mIOU'][idx], 
                                    'pixel_acc': train_metrics_historical[e]['pixel_acc'][idx], 'f1': train_metrics_historical[e]['f1'][idx], 'epoch': e}
                                json.dump(dic_line, f)
                                f.write('\n')

                            json.dump(times[e], f)
                            f.write('\n')

                            # then save the val metrics
                            dic_line = {'val_loss': val_metrics_historical['loss'][e], 'val_mIOU': val_metrics_historical['mIOU'][e], 
                                        'val_pixel_acc': val_metrics_historical['pixel_acc'][e], 
                                        'val_f1': val_metrics_historical['f1'][e], 'epoch': e}
                            json.dump(dic_line, f)
                            f.write('\n')

                            e += 1


        self.evaluator = Evaluator(self.nclass)
        self.curr_step = 0

    def training(self, epoch):
        start_time = time.time()

        self.model.train()
        tbar = tqdm(self.train_loader)

        # print("Curr Learning Rate x1: {}; Learning Rate x10: {}".format(self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[1]['lr']))
        # print("Curr Learning Rate x1: {}".format(self.optimizer.param_groups[0]['lr']))

        for i, sample in enumerate(tbar):
            image, mask, loss_weights = sample['image'], sample['mask'].long(), sample['mask_loss']

            # cuda enable image/mask
            if self.args.cuda:
                image, mask, loss_weights = image.cuda(), mask.cuda(), loss_weights.cuda()

            # need to squeeze if combined dataset
            if self.args.dataset == "combined":
                image, mask = image.squeeze(), mask.squeeze()
 
            # get output, calculate loss, perform backprop
            self.optimizer.zero_grad()
            output = self.model(image)

            if self.args.incl_bounds:
                boundary_weights = sample['boundary'].to(image.device)

                loss = self.criterion(output, mask, boundary_weights, loss_weights)
            else:
                loss = self.criterion(output, mask)

            loss.backward()
            self.optimizer.step()
            
            train_loss = loss.item()

            with torch.no_grad():
                pred = torch.nn.functional.softmax(output, dim=1)
                pred = pred.data.cpu().numpy()
                pred = np.argmax(pred, axis=1)

                target = mask.cpu().numpy()

                pixel_acc = self.evaluator.pixelAcc_manual(target, pred)
                mIOU = self.evaluator.mIOU_manual(target, pred)
                f1_score = self.evaluator.f1score_manual(target, pred)

            tbar.set_description('Train loss: %.3f' % (train_loss))
            metrics = {"train_loss": train_loss, "mIOU": mIOU, "pixel_acc": pixel_acc}
            metrics["f1"] = f1_score if f1_score is not None else 0

            self.saver.log_wandb(epoch, self.curr_step, metrics)
            self.curr_step += 1


        total_time = time.time() - start_time
        self.saver.log_wandb(None, self.curr_step, {"time" : total_time})
        
    def validation(self, epoch):
        self.model.eval()
        tbar = tqdm(self.validation_loader)
        tbar.set_description("[Epoch {}] Validation".format(epoch))

        total_loss = []
        total_pixelAcc = []
        total_mIOU = []
        total_f1 = []

        if self.args.use_wandb:
            wandb_imgs_list = []

        for i, sample in enumerate(tbar):
            image, mask, loss_weights = sample['image'], sample['mask'].long(), sample['mask_loss']
            names = sample['name']

            # cuda enable image/mask
            if self.args.cuda:
                image, mask, loss_weights = image.cuda(), mask.cuda(), loss_weights.cuda()

            # need to squeeze if combined dataset
            if self.args.dataset == "combined":
                image, mask = image.squeeze(), mask.squeeze()

            with torch.no_grad():
                output = self.model(image)

            if self.args.incl_bounds:
                boundary_weights = sample['boundary'].to(image.device)
                loss = self.criterion(output, mask, boundary_weights, loss_weights)
            else:
                loss = self.criterion(output, mask)      
                     
            total_loss.append(loss.item())

            pred = torch.nn.functional.softmax(output, dim=1)
            pred = pred.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)

            target = mask.cpu().numpy()

            total_pixelAcc.append(self.evaluator.pixelAcc_manual(target, pred))
            total_mIOU.append(self.evaluator.mIOU_manual(target, pred))
            f1 = self.evaluator.f1score_manual(target, pred)
            if f1 is not None:
                total_f1.append(f1)
            else:
                total_f1.append(0)

            # Log segmentation map to WandB
            if self.args.use_wandb:
                filename_single, image_single, pred_single, target_single = handle_concatenation(
                    self.args.dataset == "combined", None, image, pred, target, names)
                wandb_imgs_list.append(
                    wandb_segmentation_image(
                        input_img=image_single, pred_mask=pred_single, gt_mask=target_single,
                        class_labels={0: 'bg', 1: 'building'})
                )

                
        self.saver.log_wandb(epoch, self.curr_step, {
                                                        "val_loss": np.mean(total_loss),
                                                        "val_mIOU": np.mean(total_mIOU),
                                                        "val_pixel_acc": np.mean(total_pixelAcc),
                                                        "val_f1": np.mean(total_f1),
                                                    })

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

        self.saver.save_checkpoint(checkpoint, np.mean(total_loss), np.mean(total_mIOU), np.mean(total_pixelAcc), np.mean(total_f1))
        # self.saver.save_checkpoint(self.model.state_dict(), np.mean(total_loss), np.mean(total_mIOU))

        # select random image and log it to WandB
        filename, image, pred, target = handle_concatenation(
                                                             self.args.dataset == "combined",
                                                             None,
                                                             image,
                                                             pred,
                                                             target,
                                                             names
                                                         )

        #self.saver.log_wandb_image(filename, image, pred, target)

        # Log segmentations in WandB
        if self.args.use_wandb:
            wandb.log({
                'Predictions': wandb_imgs_list
            })

        self.curr_step += 1
