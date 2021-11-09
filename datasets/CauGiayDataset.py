import os
import numpy as np
from glob import glob

import torch
from torch.utils.data import Dataset


class CauGiayDataset(Dataset):
    def __init__(
        self, 
        root_dir: str, 
        boundary_kernel_size: int = None,
        transforms = None,
        stage: str ="train"
        ):

        self.stage = stage
        self.root_dir = root_dir
        self.inputs = glob(os.path.join(self.root_dir, "images", "*.npy"))

        self.transforms = transforms
        self.generate_boundary = boundary_kernel_size is not None

    def __len__(self,):
        return len(self.inputs)

    def __getitem__(self, index):
        image_filepath = self.inputs[index]
        image_filename = os.path.basename(image_filepath)

        # Load image
        image = np.load(image_filepath)
        image = torch.Tensor(image).permute(2, 0, 1)[None, :, :, :] ## Converts to 1,C,H,W

        # Apply transforms if any
        if self.transforms is not None:
            for transform in self.transforms:
                image = transform(image)

        # Convert image to [0 1] and C, H, W
        image = image.squeeze()
        image = image.float() / 255.0 # Converts image from [0 255] to [0 1] fp
        batch = {'image': image, 'name': image_filename}

        # Mask of shape H, W
        if self.stage in ["train", "test", "val"]:
            # Load masks
            mask = np.load(os.path.join(self.root_dir, "masks", image_filename.replace(".npy", "_mask.npy")))
            mask = (mask > 0).astype(np.int32)
            mask = torch.Tensor(mask)[None, None, :, :] ## 1, 1, H, W

            if self.transforms is not None:
                for transform in self.transforms:
                    mask = transform(mask)
        
            mask = torch.ceil(mask).squeeze()
            batch["mask"] = mask

            # Generate boundary if specified
            if self.generate_boundary:
                mask_wt = self.process_boundary(image_filename)
                batch['boundary'] = mask_wt
        
        return batch

    def process_boundary(self, image_filename):
        mask_wt = np.load(os.path.join(self.root_dir, "masks_wt", image_filename.replace(".npy", "_mask_wt.npy")))
        maskwt_tensor = torch.tensor(mask_wt.astype(float))

        # Convert mask weights to a scale of 0 - 1
        return maskwt_tensor / torch.max(maskwt_tensor)
