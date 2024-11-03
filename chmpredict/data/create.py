import os
import h5py
import rasterio

import numpy as np

from threading import Lock
from rasterio.enums import Resampling
from concurrent.futures import ThreadPoolExecutor

from chmpredict.data.utils import create_file_pairs, calculate_chm_mean_std

h5_lock = Lock()


def create_patch_pool(
    data_folder, output_file, patch_size=256, stride=256, num_threads=8
):
    rgb_dir = os.path.join(data_folder, "Images")
    chm_dir = os.path.join(data_folder, "CHM")

    if not os.path.isdir(rgb_dir) or not os.path.isdir(chm_dir):
        raise FileNotFoundError(f"RGB or CHM directory not found in: {data_folder}")
    if not os.listdir(rgb_dir) or not os.listdir(chm_dir):
        raise ValueError("RGB or CHM directory is empty. Please provide valid data.")

    mean_chm, std_chm = calculate_chm_mean_std(chm_dir, num_threads=8, nan_value=-9999)
    print(f"Calculated CHM Mean: {mean_chm:.4f}")
    print(f"Calculated CHM Std: {std_chm:.4f}")

    num_patches = estimate_total_patches(rgb_dir, chm_dir, patch_size, stride)

    with h5py.File(os.path.join(data_folder, output_file), "w") as f:
        rgb_dataset = f.create_dataset("rgb_patches", (num_patches, 3, patch_size, patch_size), dtype="float32")
        chm_dataset = f.create_dataset("chm_patches", (num_patches, 1, patch_size, patch_size), dtype="float32")

        # Shared index for thread-safe patch writing
        patch_idx = [0]

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for rgb_path, chm_path in create_file_pairs(rgb_dir, chm_dir):
                futures.append(
                    executor.submit(
                        process_image,
                        rgb_path,
                        chm_path,
                        rgb_dataset,
                        chm_dataset,
                        patch_idx,
                        patch_size,
                        stride,
                        mean_chm,
                        std_chm
                    )
                )

            for future in futures:
                future.result()  # This raises any exceptions caught in threads


def process_image(
    rgb_path, chm_path, rgb_dataset, chm_dataset, patch_idx, patch_size, stride, mean_chm, std_chm
):
    with rasterio.open(rgb_path) as rgb_src, rasterio.open(chm_path) as chm_src:
        rgb_resampled = (
            rgb_src.read(
                out_shape=(rgb_src.count, chm_src.height, chm_src.width),
                resampling=Resampling.bilinear,
            ).astype(np.float32)
            / 255
        )
        chm = chm_src.read(1).astype(np.float32)

        for y in range(0, chm_src.height - patch_size + 1, stride):
            for x in range(0, chm_src.width - patch_size + 1, stride):
                rgb_patch = rgb_resampled[:, y : y + patch_size, x : x + patch_size]
                chm_patch = chm[y : y + patch_size, x : x + patch_size]

                chm_patch[chm_patch == -9999] = 0  # Replace -9999 with 0 or another appropriate value

                chm_patch = (chm_patch - mean_chm) / std_chm

                chm_patch = np.expand_dims(chm_patch, axis=0)

                with h5_lock:
                    current_idx = patch_idx[0]
                    rgb_dataset[current_idx] = rgb_patch
                    chm_dataset[current_idx] = chm_patch
                    patch_idx[0] += 1


def estimate_total_patches(rgb_dir, chm_dir, patch_size, stride):
    total_patches = 0
    
    for _, chm_path in create_file_pairs(rgb_dir, chm_dir):
        with rasterio.open(chm_path) as src:
            patches_x = (src.width - patch_size) // stride + 1
            patches_y = (src.height - patch_size) // stride + 1
            total_patches += patches_x * patches_y

    return total_patches
