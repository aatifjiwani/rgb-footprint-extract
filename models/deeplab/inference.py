# https://github.com/jfzhang95/pytorch-deeplab-xception
import os
import numpy as np
from tqdm import tqdm

import torch
import pytorch_lightning as pl


class SemanticSegmentationTask(pl.LightningModule):
    def __init__(self,
                 model,
                 output_dir: str = "."):
        super().__init__()
        self.model = model
        self.output_dir = output_dir

    def forward(self, x):
        output = self.model(x)
        return output

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        images, names = batch['image'], batch['name']
        output = self.model(images)
        preds = torch.nn.functional.softmax(output, dim=1)
        preds = np.argmax(preds, axis=1)
        preds = preds.data.cpu().numpy()

        names = names.data.cpu().numpy()
        for pred, name in zip(preds, names):
            np.save(os.path.join(self.output_dir, name), pred)