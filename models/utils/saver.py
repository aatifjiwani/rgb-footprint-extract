import os
import shutil
import torch
from collections import OrderedDict
import glob
import jsonlines
import wandb
import matplotlib
import matplotlib.pyplot as plt
class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('run', args.dataset, args.checkname)
        self.save_directory = os.path.join('weights', args.checkname)
        if args.use_wandb:
            wandb.init(
                entity="<entity>",
                project="<project>",
                name=args.dataset + args.checkname,
                config=vars(args))
        
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

        self.images = 0
        self.best_loss = float('inf')
        self.best_miou = float('-inf')

    def plot_and_save_image(self, filename, input_type, image_array):
        try:
            os.makedirs(os.path.join(self.directory, '{}_{}'.format(filename, self.images)))
        except:
            pass

        matplotlib.image.imsave(os.path.join(self.directory, '{}_{}/{}.png'.format(filename, self.images, input_type)), image_array)
        return os.path.join(self.directory, "{}_{}/{}.png".format(filename, self.images, input_type))

    def log_wandb_image(self, filename, input_image, pred_mask, gt_mask):
        assert len(input_image.shape) == 3 and len(pred_mask.shape) == len(gt_mask.shape) and len(pred_mask.shape) == 2

        filename = filename.replace(".npy", "")

        input_saved = self.plot_and_save_image(filename, "input_image", input_image)
        pred_saved = self.plot_and_save_image(filename, "pred_mask", pred_mask)
        gt_saved = self.plot_and_save_image(filename, "gt_mask", gt_mask)

        if self.args.use_wandb:
            wandb.log({
                           "{}_input_{}".format(filename, self.images): wandb.Image(input_saved), 
                           "{}_pred_{}".format(filename, self.images): wandb.Image(pred_saved), 
                           "{}_gt_{}".format(filename, self.images): wandb.Image(gt_saved),
                     })

        self.images += 1


    def log_wandb(self, epoch, step, metrics):
        self.save_metrics(epoch, metrics)

        if self.args.use_wandb:
            if epoch:
                metrics["epoch"] = epoch
            wandb.log(metrics, step=step)

    def save_metrics(self, epoch, metrics):
        with jsonlines.open(os.path.join(self.directory, "metrics.jsonl"), "a") as f:
            metrics_str = "Epoch {};".format(epoch)
            for k,v in metrics.items():
                metrics_str = "{} {}: {};".format(metrics_str, k, v)

            metrics["epoch"] = epoch

            f.write(metrics)
            return metrics_str

    def save_checkpoint(self, state, val_loss, val_miou, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        if val_loss < self.best_loss:
            print("Saving best loss checkpoint")
            self.best_loss = val_loss
            torch.save(state, os.path.join(self.save_directory, 'best_loss_{}'.format(filename)))
        
        if val_miou > self.best_miou:
            print("Saving best mIOU checkpoint")
            self.best_miou = val_miou
            torch.save(state, os.path.join(self.save_directory, 'best_miou_{}'.format(filename)))
        

    def save_experiment_config(self):
        logfile = os.path.join(self.directory, 'parameters.txt')
        log_file = open(logfile, 'w')

        for key, val in vars(self.args).items():
            log_file.write(key + ':' + str(val) + '\n')

        log_file.close()
