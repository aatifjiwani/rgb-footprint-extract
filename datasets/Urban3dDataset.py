from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import numpy as np
import math
from PIL import Image

class Urban3dDataset(Dataset):
    """
        The following structure must hold

        root_dir/
            images/ (MUST BE IN .npy FORMAT) [shape H*W*C]
            masks/ (MUST BE IN .npy FORMAT) [shape H*W with entries indicating which class]
    """

    def __init__(
            self, 
            root_dir, 
            boundary_kernel_size = None,
            transforms=None,
            resize=2048,
            split=2
            ):

        self.root = root_dir
        self.filenames = [file for file in os.listdir(f"{self.root}/images/") if 'D' not in file]
        self.transforms = transforms

        self.generate_boundary = boundary_kernel_size is not None

        # num 512 images
        self.resize = resize
        self.num_512 = int(resize / 512) ** 2
        self.inputs = []
        for filename in self.filenames:
            self.inputs.extend([(filename, i*512) for i in range(self.num_512)])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        filename, index = self.inputs[index]
        assert type(filename) != list, "Data-loader only supports one request at a time"
        row_index, col_index = int(index / self.resize) * 512, index % self.resize        
  
        image = np.load(os.path.join(self.root, "images", filename))
        maskname = filename.replace(".npy", "_mask.npy")

        # slice image and mask
        image = image[row_index:row_index + 512, col_index:col_index+512, : ]
        image = torch.Tensor(image).permute(2, 0, 1)[None, :, :, :] ##Converts to 1,C,H,W

        mask = np.load(os.path.join(self.root, "masks", maskname))
        mask = (mask > 0).astype(np.int32)
        mask = mask[row_index:row_index + 512, col_index:col_index+512]
        mask = torch.Tensor(mask)[None, None, :, :] ##1, 1, H, W

        if self.transforms is not None:
            for transform in self.transforms:
                image = transform(image)
                mask = transform(mask)

        image = image.squeeze()
        image = image.float() / 255.0 # Converts image from [0 255] to [0 1] fp
        mask = torch.ceil(mask)

        batch = {'image': image, 'mask': mask.squeeze(), 'name': [filename + str(index // 512)]}

        if self.generate_boundary:
            mask_wt = self.process_boundary(filename, row_index, col_index)
            batch['boundary'] = mask_wt
        
        return batch

    def process_boundary(self, image_filename, row_index, col_index):
        mask_wt = np.load(os.path.join(self.root, "masks_wt", image_filename.replace(".npy", "_mask_wt.npy")))
        mask_wt = mask_wt[row_index:row_index + 512, col_index:col_index+512]
        maskwt_tensor = torch.tensor(mask_wt.astype(float))

        if torch.max(maskwt_tensor) > 0:
            maskwt_tensor = maskwt_tensor / torch.max(maskwt_tensor)

        # Convert mask weights to a scale of 0 - 1
        return maskwt_tensor


