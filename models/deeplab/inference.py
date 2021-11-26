# https://github.com/jfzhang95/pytorch-deeplab-xception
import os
import numpy as np

from PIL import Image

import torch
import pytorch_lightning as pl


class SemanticSegmentationTask(pl.LightningModule):
    def __init__(self,
                 model,
                 output_dir: str = ".",
                 threshold=0.0):
        super().__init__()
        self.model = model
        self.output_dir = output_dir
        self.threshold = threshold

    def forward(self, x):
        output = self.model(x)
        return output

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        images, names = batch['image'], batch['name']
        output = self.model(images)
        preds = torch.nn.functional.softmax(output, dim=1)

        confidence = torch.max(preds, dim=1, keepdim=False)[0]
        confidence = confidence > self.threshold
        confidence = confidence.float()
        
        labels = torch.argmax(preds, axis=1)
        labels = labels * confidence

        labels = labels.data.cpu().numpy()
        for label, name in zip(labels, names):
            im = Image.fromarray(label.astype(np.uint8) * 255)
            im.save(os.path.join(self.output_dir, name))