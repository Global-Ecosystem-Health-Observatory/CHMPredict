import os
import rasterio
import numpy as np

from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

sum_lock = Lock()  # Lock for thread-safe operations


def create_file_pairs(rgb_dir, chm_dir):
    rgb_files = [os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) if f.endswith('.jp2')]
    chm_files = [os.path.join(chm_dir, f) for f in os.listdir(chm_dir) if f.endswith('.tif')]

    rgb_dict = {}
    chm_dict = {}

    for f in rgb_files:
        identifier = os.path.splitext(os.path.basename(f))[0]
        rgb_dict[identifier] = f

    for f in chm_files:
        identifier = os.path.basename(f)[4:-9]
        chm_dict[identifier] = f

    file_pairs = [(rgb_dict[id], chm_dict[id]) for id in rgb_dict.keys() if id in chm_dict]

    return file_pairs


def process_chm_file(chm_path, nan_value=-9999):
    """Process a single CHM file to calculate sum, sum of squares, and count, excluding NaN values."""
    with rasterio.open(chm_path) as chm_src:
        chm_data = chm_src.read(1).astype(np.float32)
        
        valid_data = chm_data[chm_data != nan_value]
        
        file_sum = valid_data.sum()
        file_sum_sq = np.sum(valid_data ** 2)
        file_pixels = valid_data.size
    
    return file_sum, file_sum_sq, file_pixels


def calculate_chm_mean_std(chm_dir, num_threads=8, nan_value=-9999):
    total_sum = 0
    total_sum_sq = 0
    total_pixels = 0

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        
        for chm_file in os.listdir(chm_dir):
            if chm_file.endswith(".tif"):
                chm_path = os.path.join(chm_dir, chm_file)
                futures.append(executor.submit(process_chm_file, chm_path, nan_value))
        
        for future in as_completed(futures):
            file_sum, file_sum_sq, file_pixels = future.result()
            
            with sum_lock:
                total_sum += file_sum
                total_sum_sq += file_sum_sq
                total_pixels += file_pixels

    mean = total_sum / total_pixels
    std = np.sqrt(total_sum_sq / total_pixels - mean ** 2)
    
    return mean, std