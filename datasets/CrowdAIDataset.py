
import os
import torch
import skimage.io as io
import numpy as np

from torch.utils.data import Dataset

class CrowdAIDataset(Dataset):
    def __init__(
            self, 
            root_dir, 
            boundary_kernel_size = None,
            transforms=None,
            data_slice=None,
            ):

        self.root_dir = root_dir
        self.inputs = list(os.listdir(f"{self.root_dir}/images/"))

        # Temporary strategy
        np.random.shuffle(self.inputs)
        if data_slice is not None:
            self.inputs = self.inputs[: int(len(self.inputs) * data_slice) ]

        self.transforms = transforms

        self.generate_boundary = boundary_kernel_size is not None

    def __len__(self,):
        return len(self.inputs)

    def __getitem__(self, index):
        image_filename = self.inputs[index]

        # Load image
        image = io.imread(os.path.join(self.root_dir, "images", image_filename))
        image = torch.Tensor(image).permute(2, 0, 1)[None, :, :, :] ##Converts to 1,C,H,W

        # Load masks
        mask = np.load(os.path.join(self.root_dir, "masks", image_filename + "_mask.npy"))
        mask = (mask > 0).astype(np.int32)
        mask = torch.Tensor(mask)[None, None, :, :] ##1, 1, H, W

        # Apply transforms if any
        if self.transforms is not None:
            for transform in self.transforms:
                image = transform(image)
                mask = transform(mask)

        # Convert image to [0 1] and C, H, W
        image = image.squeeze()
        image = image.float() / 255.0 # Converts image from [0 255] to [0 1] fp

        # Mask of shape H, W
        mask = torch.ceil(mask).squeeze()

        batch = {'image': image, 'mask': mask, 'name': [image_filename.replace(".jpg", "")]}

        # Generate boundary if specified
        if self.generate_boundary:
            mask_wt = self.process_boundary(image_filename)
            batch['boundary'] = mask_wt

        return batch

    def process_boundary(self, image_filename):
        mask_wt = np.load(os.path.join(self.root_dir, "masks_wt", image_filename + "_mask_wt.npy"))
        maskwt_tensor = torch.tensor(mask_wt.astype(float))

        if torch.max(maskwt_tensor) > 0:
            maskwt_tensor = maskwt_tensor / torch.max(maskwt_tensor)

        # Convert mask weights to a scale of 0 - 1
        return maskwt_tensor 
        