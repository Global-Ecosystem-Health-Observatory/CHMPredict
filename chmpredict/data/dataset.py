import torch
import rasterio
import numpy as np

from torch.utils.data import Dataset
from rasterio.enums import Resampling
from sklearn.model_selection import train_test_split

from chmpredict.data.utils import create_file_pairs


class CHMDataset(Dataset):
    def __init__(self, rgb_dir, chm_dir, patch_size=256, transform=None, mode="train", test_size=0.2, val_size=0.25, random_state=42):
        self.rgb_dir = rgb_dir
        self.chm_dir = chm_dir
        self.patch_size = patch_size
        self.transform = transform
        self.mode = mode

        self.file_pairs = create_file_pairs(rgb_dir, chm_dir)
        
        train_val_pairs, test_pairs = train_test_split(self.file_pairs, test_size=test_size, random_state=random_state)
        train_pairs, val_pairs = train_test_split(train_val_pairs, test_size=val_size, random_state=random_state)

        if mode == "train":
            self.files = train_pairs
        elif mode == "val":
            self.files = val_pairs
        elif mode == "test":
            self.files = test_pairs
        else:
            raise ValueError("Mode must be 'train', 'val', or 'test'")        

    def __len__(self):
        return len(self.files) * 100  # Adjust based on sampling needs

    def __getitem__(self, idx):
        rgb_path, chm_path = self.files[idx % len(self.files)]
        
        with rasterio.open(rgb_path) as rgb_src, rasterio.open(chm_path) as chm_src:
            rgb_resampled = rgb_src.read(
                out_shape=(rgb_src.count, chm_src.height, chm_src.width),
                resampling=Resampling.bilinear
            ).astype(np.float32) / 255

            chm = chm_src.read(1).astype(np.float32)

            x = np.random.randint(0, chm_src.width - self.patch_size + 1)
            y = np.random.randint(0, chm_src.height - self.patch_size + 1)
            
            rgb_patch = rgb_resampled[:, y:y+self.patch_size, x:x+self.patch_size]
            chm_patch = chm[y:y+self.patch_size, x:x+self.patch_size]
        
        if chm_patch.max() - chm_patch.min() > 0:
            chm_patch = (chm_patch - chm_patch.min()) / (chm_patch.max() - chm_patch.min())
        else:
            chm_patch = np.zeros_like(chm_patch)

        rgb_tensor = torch.from_numpy(rgb_patch)
        chm_tensor = torch.from_numpy(chm_patch).unsqueeze(0)

        if self.transform:
            rgb_tensor = self.transform(rgb_tensor)
            chm_tensor = self.transform(chm_tensor)

        return rgb_tensor, chm_tensor