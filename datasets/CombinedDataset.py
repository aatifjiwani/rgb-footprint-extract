import sys
import os
import torch
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset

from datasets import Urban3dDataset, SpaceNetDataset, CrowdAIDataset
from models.utils.custom_transforms import tensor_resize
from models.utils.collate_fn import generate_split_collate_fn

class CombinedDataset(Dataset):
    def __init__(self, data_root, boundary=False, train=True):
        self.mode = "train" if train else "val"
        self.use_boundary = boundary

        # Urban3d / SpaceNet transform
        us_transform = [tensor_resize(512)]

        # AI Crowd transform
        ai_transform = [tensor_resize(256)]

        # Initialize datasets
        self.data_root = data_root
        self.urban3d = Urban3dDataset(os.path.join(self.data_root, "Urban3D/", self.mode), boundary, False, None)
        self.spaceNet = SpaceNetDataset(os.path.join(self.data_root, "SpaceNet/Vegas/", self.mode), boundary, us_transform) 
        self.aiCrowd = CrowdAIDataset(os.path.join(self.data_root, "AICrowd/", self.mode), boundary, ai_transform, data_slice=0.1)

        """
        Urban3D -> 2048 x 2048 -> 512 x 512 -> 4 splits of 256 x 256 (64) 
        SpaceNet -> 650 x 650 -> 512 x 512 -> 4 splits of 256 x 256 (4)
        CrowdAI -> 300 x 300 -> 256 x 256
        """
        
        # Input list
        self.inputs = [self.urban3d]*len(self.urban3d) + [self.spaceNet]*len(self.spaceNet) 
        
        # Truncate CrowdAI to use batch size of 4
        self.ai_len = 4
        len_ai = int(np.ceil(len(self.aiCrowd) / self.ai_len)) # Train with batches of 4
        self.inputs += [self.aiCrowd]*len_ai

        np.random.shuffle(self.inputs)

        self.urban3d_iter = iter(self.urban3d)
        self.spaceNet_iter = iter(self.spaceNet)
        self.aiCrowd_iter = iter(self.aiCrowd)

        self.split_fn = generate_split_collate_fn(512, 2, self.use_boundary)

        print("Combined Dataset with mode {}".format(self.mode))
        print("CrowdAI: {}; Urban3D: {}; SpaceNet: {}".format(len_ai, len(self.urban3d), len(self.spaceNet)))

    def __len__(self,):
        return len(self.inputs)

    def __getitem__(self, index):
        curr_dataset = self.inputs[index]

        if curr_dataset == self.urban3d:  # urban3d
            inputs = next(self.urban3d_iter)
            inputs["name"][0] += "_urban3d"
            batch = self.split_fn([inputs])
        elif curr_dataset == self.spaceNet:  #space net
            inputs = next(self.spaceNet_iter)
            inputs["name"][0] += "_spaceNet"
            batch = self.split_fn([inputs])
        else:  # ai crowd
            samples = []
            for _ in range(self.ai_len):
                next_input = next(self.aiCrowd_iter)
                next_input["name"][0] += "_crowdAI"
                samples.append(next_input)

            batch = self.combined_collate_fn(samples)

        return batch

    def combined_collate_fn(self, samples):
        images, masks, names, boundaries = [], [], [], []
        for sample in samples:
            images.append(sample["image"])
            masks.append(sample["mask"])
            names.append(sample["name"])

            if self.use_boundary:
                boundaries.append(sample["boundary"])

        batch = {'image': torch.stack(images, axis=0), 'mask': torch.stack(masks, axis=0), 'name': names}
        if self.use_boundary:
            batch["boundary"] = torch.stack(boundaries, axis=0)

        return batch

