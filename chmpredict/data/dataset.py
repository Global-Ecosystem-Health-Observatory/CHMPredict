import h5py
import torch

import numpy as np

from torch.utils.data import Dataset
from PIL import Image


class CHMDataset(Dataset):
    def __init__(
        self,
        hdf5_file,
        transform=None,
        mode="train",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    ):
        self.hdf5_file = hdf5_file
        self.transform = transform
        self.mode = mode

        with h5py.File(hdf5_file, "r") as f:
            self.num_patches = f["rgb_patches"].shape[0]

        train_end = int(train_ratio * self.num_patches)
        val_end = train_end + int(val_ratio * self.num_patches)

        if mode == "train":
            self.indices = range(0, train_end)
        elif mode == "val":
            self.indices = range(train_end, val_end)
        elif mode == "test":
            self.indices = range(val_end, self.num_patches)
        else:
            raise ValueError("Mode must be 'train', 'val', or 'test'")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        hdf5_index = self.indices[idx]

        with h5py.File(self.hdf5_file, "r") as f:
            rgb_patch = f["rgb_patches"][hdf5_index]  # Shape: (3, patch_size, patch_size)
            chm_patch = f["chm_patches"][hdf5_index]  # Shape: (1, patch_size, patch_size)

        rgb_patch = rgb_patch.transpose(1, 2, 0)  # Convert from CHW to HWC
        rgb_patch = (rgb_patch * 255).astype(np.uint8)
        rgb_patch = Image.fromarray(rgb_patch)

        if self.transform:
            rgb_patch = self.transform(rgb_patch)  # Transform now works with PIL image

        chm_tensor = torch.from_numpy(chm_patch).float()  # Shape: (1, patch_size, patch_size)

        return rgb_patch, chm_tensor
