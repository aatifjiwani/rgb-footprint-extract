import os
import numpy as np
import torch
from torch.nn.functional import interpolate
from tqdm import tqdm
from skimage.segmentation import find_boundaries

class Processor():
    def __init__(self, root_path, w0, sigma, inc, resize=None, start=None):
        self.root_path = root_path
        self.masks_path = os.path.join(self.root_path, "masks")

        self.masks_wt_path = os.path.join(self.root_path, "masks_wt")
        if not os.path.exists(self.masks_wt_path):
            os.mkdir(self.masks_wt_path)
        

        self.w0, self.sigma = w0, sigma
        self.inc = inc
        self.resize = resize
        self.start = start

    def process(self,):
        pbar = tqdm(os.listdir(self.masks_path))
        if self.start is not None:
            print("Starting at {}".format(self.start))
            pbar = tqdm(list(os.listdir(self.masks_path))[self.start:])

        for mask_path in pbar:
            mask = np.load(os.path.join(self.masks_path, mask_path))
            mask = (mask > 0).astype(np.int32)

            if self.resize is not None:
                mask = self.resize_mask(mask).squeeze().astype(np.int32)

            mask_weight = np.expand_dims(mask, axis=0)
            endpoint = mask.shape[0]
            max_step = int(np.ceil(endpoint / self.inc))

            for i in range(max_step):
                si, ei = i*self.inc, min(endpoint, i*self.inc+self.inc)

                for j in range(max_step):
                    sj, ej = j*self.inc, min(endpoint, j*self.inc+self.inc)

                    if len(np.unique(mask[si:ei, sj:ej])) > 1:
                        mask_weight[:, si:ei, sj:ej] = self.make_weight_map(mask_weight[:, si:ei, sj:ej])
                    else:
                        mask_weight[:, si:ei, sj:ej] = 0

            np.save(
                os.path.join(self.masks_wt_path, mask_path.replace("_mask.npy", "_mask_wt.npy")),
                mask_weight.astype(np.uint8).squeeze()
            )

    def resize_mask(self, mask):
        return interpolate(torch.tensor(mask).unsqueeze(0).unsqueeze(0).float(), size=self.resize, mode="nearest").detach().numpy()

    def make_weight_map(self, masks):
        """
        Generate the weight maps as specified in the UNet paper
        for a set of binary masks.
        
        Parameters
        ----------
        masks: array-like
            A 3D array of shape (n_masks, image_height, image_width),
            where each slice of the matrix along the 0th axis represents one binary mask.

        Returns
        -------
        array-like
            A 2D array of shape (image_height, image_width)
        
        """
        nrows, ncols = masks.shape[1:]
        masks = (masks > 0).astype(int)
        distMap = np.zeros((nrows * ncols, masks.shape[0]))
        X1, Y1 = np.meshgrid(np.arange(nrows), np.arange(ncols))
        X1, Y1 = np.c_[X1.ravel(), Y1.ravel()].T
        for i, mask in enumerate(masks):
            # find the boundary of each mask,
            # compute the distance of each pixel from this boundary
            bounds = find_boundaries(mask, mode='inner')
            X2, Y2 = np.nonzero(bounds)
            xSum = (X2.reshape(-1, 1) - X1.reshape(1, -1)) ** 2
            ySum = (Y2.reshape(-1, 1) - Y1.reshape(1, -1)) ** 2
            distMap[:, i] = np.sqrt(xSum + ySum).min(axis=0)
        ix = np.arange(distMap.shape[0])
        if distMap.shape[1] == 1:
            d1 = distMap.ravel()
            border_loss_map = self.w0 * np.exp((-1 * (d1) ** 2) / (2 * (self.sigma ** 2)))
        else:
            if distMap.shape[1] == 2:
                d1_ix, d2_ix = np.argpartition(distMap, 1, axis=1)[:, :2].T
            else:
                d1_ix, d2_ix = np.argpartition(distMap, 2, axis=1)[:, :2].T
            d1 = distMap[ix, d1_ix]
            d2 = distMap[ix, d2_ix]
            border_loss_map = self.w0 * np.exp((-1 * (d1 + d2) ** 2) / (2 * (self.sigma ** 2)))
        xBLoss = np.zeros((nrows, ncols))
        xBLoss[X1, Y1] = border_loss_map
        # class weight map
        loss = np.zeros((nrows, ncols))
        w_1 = 1 - masks.sum() / loss.size
        w_0 = 1 - w_1
        loss[masks.sum(0) == 1] = w_1
        loss[masks.sum(0) == 0] = w_0
        ZZ = xBLoss + loss
        return ZZ

if __name__ == "__main__":
    """

    Example Usage:
        processor = Processor("/data/AICrowd/train/", 10, 7.5, 150)
        processor.process()

    """
    pass
