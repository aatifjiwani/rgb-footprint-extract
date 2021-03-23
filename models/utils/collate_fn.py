import numpy as np
import torch 
import os

import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def generate_split_collate_fn(original_size, factor, incl_boundary = False):

    def split_grid_collate_fn(current_batch):

        # split image/mask into blocks of size
        def scrolling_window(image, mask, size, boundary=None):
            split_images, split_masks = [], []
            split_boundaries = []

            curr_size = image.shape[1]
            iterations = curr_size // size

            for x_stride in range(iterations):
                for y_stride in range(iterations):
                    
                    x_bound_1, x_bound_2 = (x_stride*size), (x_stride+1)*size
                    y_bound_1, y_bound_2 = (y_stride*size), (y_stride+1)*size

                    split_images.append( image[:, x_bound_1:x_bound_2, y_bound_1:y_bound_2] )
                    split_masks.append( mask[x_bound_1:x_bound_2, y_bound_1:y_bound_2] )

                    if boundary is not None:
                        split_boundaries.append( boundary[x_bound_1:x_bound_2, y_bound_1:y_bound_2] )

            return split_images, split_masks, split_boundaries

        stacked_split_samples = []
        stacked_split_masks = []
        stacked_split_boundaries = []
        all_filenames = []

        for batch in current_batch:
            sample, masks, name = batch['image'], batch['mask'], batch['name']
            if incl_boundary:
                boundary = batch['boundary']
                split_sample, split_mask, split_boundary = scrolling_window(sample, masks, original_size // factor, boundary=boundary) # List of N elements of size C, 512, 512
                stacked_split_boundaries.append(torch.stack(split_boundary))
            else:
                split_sample, split_mask, _ = scrolling_window(sample, masks, original_size // factor) # List of N elements of size C, 512, 512

            stacked_split_samples.append( torch.stack(split_sample) ) # create tensor (N, C, 512, 512)
            stacked_split_masks.append( torch.stack(split_mask) ) # create tensor (N, 512, 512)

            all_filenames += name * len(split_sample) # duplicate the file name. Could be used for matching purposes
            
        
        collate_batch = {
            'image': torch.cat(stacked_split_samples), 
            'mask': torch.cat(stacked_split_masks), 
            'name': all_filenames
        }

        if incl_boundary:
            collate_batch['boundary'] = torch.cat(stacked_split_boundaries)

        return collate_batch

    return split_grid_collate_fn

def handle_concatenation(is_combined, is_split, image, pred, target, names):
    if is_combined:
        test_name = names[0][0] #List of 1 len tuples
        print(test_name)
        if "urban3d" in test_name or "spaceNet" in test_name:
            names = [n[0] if type(n) == tuple else n for n in names ]
            return handle_splits(2, image, pred, target, names)
        elif "crowdAI" in test_name[0]: # or "spaceNet" in test_name[0]:
            names = [n[0][0] for n in names]
            return handle_normal(image, pred, target, names)
        else:
            raise Exception("something went wrong in handle_concat or data loading from combined")
    elif is_split is not None:
        return handle_splits(is_split, image, pred, target, names)
    else:
        return handle_normal(image, pred, target, names)

def handle_normal(image, pred, target, names):
    random_index = np.random.choice(image.shape[0], 1).item()
        
    filename = names[0][random_index] # names is a list of tuples with the list len being 1 and the tuple len being the batch size
    image = image[random_index, :, :, :].permute(1, 2, 0).cpu().numpy() # C, H, W
    pred = pred[random_index, :, :] # H, W
    target = target[random_index, :, :] #H, W

    return filename, image, pred, target

def handle_splits(split_factor, image, pred, target, names):
    num_splits = split_factor ** 2
    random_index = np.random.choice(image.shape[0]//num_splits, 1).item()

    random_index_adj = random_index * num_splits
    filename = names[random_index_adj]
    split_image = image[random_index_adj:random_index_adj+num_splits,:,:,:].cpu().numpy()
    split_pred = pred[random_index_adj:random_index_adj+num_splits,:,:]
    split_target = target[random_index_adj:random_index_adj+num_splits, :, :]

    image = concatenate_images(split_image, split_factor)
    pred = concatenate_images(split_pred, split_factor)
    target = concatenate_images(split_target, split_factor)

    return filename, image, pred, target

def concatenate_images(split_tensor, split_factor):
    """
    input: n-split tensor where n is a power of 2
    4, 256, 256
        w - 256, 256
        x - 256, 256
        y - 256, 256
        z - 256
    Desired
    [
        w x 
        y z
    ]

    2D array X (n, x)
    2D array Y (n, y)

    [X Y] (n, x+y)

    2D array Z (x, x)
    [Z Z]
    [Z Z]
    create array of shape (2X, 2X)

    Traditional:
    np.reshape(input, 512, 512)

    """

    assert len(split_tensor.shape) == 3 or len(split_tensor.shape) == 4
    assert split_tensor.shape[0] == split_factor**2

    num_splits = split_tensor.shape[0]
    splits = np.split(split_tensor, num_splits, axis=0)

    joined_image = np.block([splits[i:i+split_factor] for i in range(0, num_splits, split_factor)]).squeeze()
    if len(joined_image.shape) == 3:
        assert split_tensor.shape[-1] * split_factor == joined_image.shape[-1]
        joined_image = np.transpose(joined_image, (1,2,0))
    else:
        assert len(joined_image.shape) == 2 or len(joined_image.shape) == 3
        assert split_tensor.shape[1] * split_factor == joined_image.shape[0]

    return joined_image


def save_and_test_images(sample, prefix):
    print(os.getcwd())
    numpy_sample = sample.detach().numpy()
    for i, split_img in enumerate(numpy_sample):
        print(split_img.shape)
        print(split_img.max())
        print(split_img.min())
        matplotlib.image.imsave('models/utils/testing/{}_{}.png'.format(prefix, i), split_img)


