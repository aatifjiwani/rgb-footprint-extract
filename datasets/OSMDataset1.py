import os
import numpy as np
import torch

from torch.utils.data import Dataset

class OSMDataset1(Dataset):
    def __init__(
        self, 
        root_dir, 
        boundary_kernel_size = None,
        transforms=None,
        ):

        self.root_dir = root_dir
        self.inputs = list(sorted(os.listdir(f"{self.root_dir}/images/")))
        self.inputs = [f'{i}__0' for i in self.inputs] + [f'{i}__1' for i in self.inputs] + [f'{i}__2' for i in self.inputs] + [f'{i}__3' for i in self.inputs]
        self.transforms = transforms

        self.generate_boundary = boundary_kernel_size is not None

    def __len__(self,):
        return len(self.inputs)

    def __getitem__(self, index):
        image_filename = self.inputs[index].split('__')[0]
        segment = int(self.inputs[index].split('__')[1])

        # Load image
        image = None
        mask_loss = None
        image = np.load(os.path.join(self.root_dir, "images", image_filename))
        mask_loss = np.load(os.path.join(self.root_dir, "masks", image_filename.replace(".npy", "_mask.npy")))
        if segment == 0:
            image = image[:1024, :1024, :]
            mask_loss = mask_loss[:1024, :1024]
        elif segment == 1:
            image = image[:1024, 1024:, :]
            mask_loss = mask_loss[:1024, 1024:]
        elif segment == 2:
            image = image[1024:, :1024, :]
            mask_loss = mask_loss[1024:, :1024]
        elif segment == 3:
            image = image[1024:, 1024:, :]
            mask_loss = mask_loss[1024:, 1024:]


        image = torch.Tensor(image).permute(2, 0, 1)[None, :, :, :] ##Converts to 1,C,H,W -- the NAIP imagery is RGBA, so need to index up to 3

        # Load masks (CHANGED NAMING OF MASK to MASK_LOSS FOR PHASE 2)
        mask = (mask_loss > 0).astype(np.int32)
        mask = torch.Tensor(mask)[None, None, :, :] ##1, 1, H, W

        # EXCLUSIVE TO PHASE 2: Include loss weights
        # We pass in raw mask_loss but we will process this in loss.py
        mask_loss = torch.Tensor(mask_loss)[None, None, :, :]

        # Apply transforms if any
        if self.transforms is not None:
            for transform in self.transforms:
                image = transform(image)
                mask = transform(mask)
                mask_loss = transform(mask_loss)

        # Convert image to [0 1] and C, H, W
        image = image.squeeze()
        image = image.float() / 255.0 # Converts image from [0 255] to [0 1] fp

        # Mask of shape H, W
        mask = torch.ceil(mask).squeeze()

        batch = {'image': image, 'mask': mask, 'mask_loss': mask_loss, 'name': [f'{image_filename.replace(".npy", "")}_{segment}']}

        # Generate boundary if specified
        if self.generate_boundary:
            mask_wt = self.process_boundary(image_filename, segment)
            batch['boundary'] = mask_wt
        
        return batch

    def process_boundary(self, image_filename, segment):
        mask_wt = np.load(os.path.join(self.root_dir, "masks_wt", image_filename.replace(".npy", "_mask_wt.npy")))
        if segment == 0:
            mask_wt = mask_wt[:1024, :1024]
        elif segment == 1:
            mask_wt = mask_wt[:1024, 1024:]
        elif segment == 2:
            mask_wt = mask_wt[1024:, :1024]
        elif segment == 3:
            mask_wt = mask_wt[1024:, 1024:]
        maskwt_tensor = torch.tensor(mask_wt.astype(float))

        # Convert mask weights to a scale of 0 - 1
        return maskwt_tensor / torch.max(maskwt_tensor)
        