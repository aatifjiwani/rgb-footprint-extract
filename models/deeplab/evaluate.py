# https://github.com/jfzhang95/pytorch-deeplab-xception
import os
import numpy as np
from tqdm import tqdm

import torch
import pytorch_lightning as pl


class SemanticSegmentationTask(pl.LightningModule):
    def __init__(self, model):
        super().__init__()

        self.model = model

    def forward(self, x):
        output = self.model(x)
        return output

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        image = batch['image']
        output = self.model(image)
        preds = torch.nn.functional.softmax(output, dim=1)
        preds = np.argmax(preds, axis=1)
        # preds = preds.data.cpu().numpy()

        return preds