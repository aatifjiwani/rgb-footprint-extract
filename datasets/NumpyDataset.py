import numpy as np
import torch
from torch.utils.data.dataset import Dataset

class NumpyDataset(Dataset):

    """ A dataset for single numpy arrays that themselves represent satellite imagery. 
    The arrays must be in shape (H, W, C) and each pixel must be in range 0 - 255
    """

    def __init__(self, filename, window_size=None, stride=None, transforms=None):
        """

        Args:
            filename: The path to the numpy array
            window_size : If the numpy array to too big to fit onto the GPU, then perform grid slicing where each grid block is of size `window_size`
            stride: The stride at which to perform grid slicing. `window_size` and `stride` must be set together. 
            transform: Transforms to apply to each image.

        If `window_size` doesn't divide the height of the array evenly (which is what is likely to happen) then we will sample an additional row of blocks that are aligned to the bottom of the array.
        We do a similar operation if `window_size` doesn't divide the width of the array evenly -- by appending an additional column.
        """

        self.filename = filename
        self.transforms = transforms
        self.is_grid_sampling = window_size and stride
        self.window_size = window_size

        np_array = np.load(self.filename, mmap_mode='r')
        self.height, self.width, _ = np_array.shape

        self.grid_coordinates, self.num_points = None, 1
        if self.is_grid_sampling:
            self.grid_coordinates = [] # upper left coordinate (y,x), of each block that this Dataset will return

            for y in set(list(range(0, self.height - self.window_size, stride)) + [self.height - self.window_size]):
                for x in set(list(range(0, self.width - self.window_size, stride)) + [self.width - self.window_size]):
                    self.grid_coordinates.append((y,x))
            self.num_points = len(self.grid_coordinates)

    def __len__(self):
        return self.num_points

    def __getitem__(self, index):
        image = np.load(self.filename)
        if self.is_grid_sampling:
            row, col = self.grid_coordinates[index]
            image = image[row:row+self.window_size, col:col+self.window_size, :]
        
        image = torch.Tensor(image).permute(2, 0, 1)
        if self.transforms is not None:
            for transform in self.transforms:
                image = transform(image)

        return {
            "image": image.float() / 255.0, 
            "coord": np.array(self.grid_coordinates[index]) if self.is_grid_sampling else (0, 0)
        }
