import os
import numpy as np
import torch

from torch.utils.data import Dataset

class CombinedDataset_NAIP(Dataset):
    def __init__(
        self, 
        root_dir, 
        data_type,
        boundary_kernel_size = None,
        transforms=None,
        ):

        self.root_dir = root_dir
        self.data_type = data_type

        sj = list(sorted(os.listdir(os.path.join(self.root_dir, 'san_jose_naip_512', self.data_type, 'images'))))
        la = list(sorted(os.listdir(os.path.join(self.root_dir, 'los_angeles_naip/2016_rgb_footprint_512', self.data_type, 'images'))))
        self.dic = {'sj': sj, 'la': la}

        self.inputs = la+sj
        self.transforms = transforms

        self.generate_boundary = boundary_kernel_size is not None

    def __len__(self,):
        return len(self.inputs)

    def __getitem__(self, index):
        image_filename = self.inputs[index]

        # get sub_dir
        fp = None
        if image_filename in self.dic['sj']:
            fp = os.path.join(self.root_dir, 'san_jose_naip_512', self.data_type)
        else:
            fp = os.path.join(self.root_dir, 'los_angeles_naip/2016_rgb_footprint_512', self.data_type)

        # Load image
        image = np.load(os.path.join(fp, "images", image_filename))
        image = torch.Tensor(image).permute(2, 0, 1)[None, :3, :, :] ##Converts to 1,C,H,W -- the NAIP imagery is RGBA, so need to index up to 3

        # Load masks
        mask = np.load(os.path.join(fp, "masks", image_filename.replace(".npy", "_mask.npy")))
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

        batch = {'image': image, 'mask': mask, 'name': [image_filename.replace(".npy", "")]}

        # Generate boundary if specified
        if self.generate_boundary:
            mask_wt = self.process_boundary(image_filename, fp)
            batch['boundary'] = mask_wt
        
        return batch

    def process_boundary(self, image_filename, fp):
        mask_wt = np.load(os.path.join(fp, "masks_wt", image_filename.replace(".npy", "_mask_wt.npy")))
        maskwt_tensor = torch.tensor(mask_wt.astype(float))

        # Convert mask weights to a scale of 0 - 1
        return maskwt_tensor / torch.max(maskwt_tensor)
        