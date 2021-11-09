import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .CrowdAIDataset import CrowdAIDataset
from .Urban3dDataset import Urban3dDataset
from .SpaceNetDataset import SpaceNetDataset
from .CombinedDataset import CombinedDataset
from .CauGiayDataset import CauGiayDataset

def build_dataloader(dataset, data_root, boundary_ks, transforms, resize=2048, split=2):
    if dataset == "urban3d":
        train =  Urban3dDataset(
            os.path.join(data_root, "Urban3D", "train"), 
            boundary_kernel_size=boundary_ks, 
            transforms=transforms,
            resize=resize, split=split
        )
        val = Urban3dDataset(
            os.path.join(data_root, "Urban3D", "val"), 
            boundary_kernel_size=boundary_ks, 
            transforms=transforms,
            resize=resize, split=split
        )
    elif dataset == "spaceNet":
        train = SpaceNetDataset(os.path.join(data_root, "SpaceNet", "Vegas", "train"), boundary_ks, transforms)
        val = SpaceNetDataset(os.path.join(data_root, "SpaceNet", "Vegas", "val"), boundary_ks, transforms)
    elif dataset == "cauGiay":
        train = CauGiayDataset(os.path.join(data_root, "CauGiay", "train"), boundary_ks, transforms)
        val = CauGiayDataset(os.path.join(data_root, "CauGiay", "val"), boundary_ks, transforms)
    elif dataset == "crowdAI":
        train = CrowdAIDataset(os.path.join(data_root, "AICrowd", "train"), boundary_ks, transforms, data_slice=0.15)
        val = CrowdAIDataset(os.path.join(data_root, "AICrowd", "val"), boundary_ks, transforms, data_slice=0.15)
    elif dataset == "combined":
        train = CombinedDataset(data_root, boundary=(boundary_ks is not None))
        val = CombinedDataset(data_root, boundary=(boundary_ks is not None), train=False)
    else:
        raise NotImplementedError()

    return train, val

def build_test_dataloader(dataset, data_root, transforms):
    if dataset == "urban3d":
        return Urban3dDataset(os.path.join(data_root, "Urban3D/test"), boundary_kernel_size=None, transforms=transforms)
    elif dataset == "spaceNet":
        return SpaceNetDataset(os.path.join(data_root, "SpaceNet/Vegas/test"), None, transforms)
    elif dataset == "crowdAI":
        return CrowdAIDataset(os.path.join(data_root, "AICrowd/test"), None, transforms)
    else:
        raise NotImplementedError()


class DeepLabDataModule(pl.LightningDataModule):
    def __init__(self,
                 args
                 ):
        super().__init__()

        self.dataset = args.dataset
        self.data_root = args.data_root
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.workers = args.workers

        # Define transforms and Dataloader
        self.boundary_ks = args.bounds_kernel_size if args.incl_bounds else None
        self.deeplab_collate_fn = None
        self.transform = None

    def prepare_data(self):
        # called only on 1 GPU
        pass

    def setup(self,
              stage: str = "fit"):
        resize = 2048

        if stage == "fit" or stage is None:
            split = 2
            self.train_dataset, self.val_dataset = build_dataloader(self.dataset,
                                                                    self.data_root,
                                                                    self.boundary_ks,
                                                                    self.transform,
                                                                    resize,
                                                                    split)
        
        if stage == "inference" or stage is None:
            self.inference_dataset = None

    def train_dataloader(self):
        if self.train_dataset is not None:
            return DataLoader(
                    self.train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.workers,
                    collate_fn=self.deeplab_collate_fn
                )

    def val_dataloader(self):
        if self.val_dataset is not None:
            return DataLoader(
                    self.val_dataset,
                    batch_size=self.test_batch_size,
                    shuffle=False,
                    num_workers=self.workers,
                    collate_fn=self.deeplab_collate_fn
                )
    
    def predict_dataloader(self):
        if self.inference_dataset is not None:
            return DataLoader(
                    self.inference_dataset,
                    batch_size=self.test_batch_size,
                    shuffle=False,
                    num_workers=self.workers,
                    collate_fn=self.deeplab_collate_fn
                )
