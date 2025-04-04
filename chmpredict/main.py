import os
import torch
import configargparse

from chmpredict.data.loader import load_fn
from chmpredict.data.create import create_patch_pool
from chmpredict.data.utils import calculate_chm_mean_std

from chmpredict.model.build import build_fn
from chmpredict.model.callback import EarlyStopping, ModelCheckpoint
from chmpredict.model.train import train_fn
from chmpredict.model.eval import eval_fn


def main(config):
    print("Starting CHM Predictor Process...")

    rgb_dir = os.path.join(config.data_folder, "Images")
    chm_dir = os.path.join(config.data_folder, "CHM")

    if not os.path.isdir(rgb_dir) or not os.path.isdir(chm_dir):
        raise FileNotFoundError(f"RGB or CHM directory not found in: {config.data_folder}")
    if not os.listdir(rgb_dir) or not os.listdir(chm_dir):
        raise ValueError("RGB or CHM directory is empty. Please provide valid data.")

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    hdf5_path = os.path.join(config.data_folder, config.hdf5_file)

    mean_chm, std_chm = calculate_chm_mean_std(chm_dir, num_threads=8, nan_value=-9999)
    print(f"Calculated CHM Mean: {mean_chm:.4f}")
    print(f"Calculated CHM Std: {std_chm:.4f}")

    if not os.path.exists(hdf5_path):
        print(f"Creating patch pool at {hdf5_path}...")
        create_patch_pool(rgb_dir, chm_dir, hdf5_path, mean_chm, std_chm, patch_size=256, stride=256)
    else:
        print(f"HDF5 file {hdf5_path} already exists. Skipping creation.")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    _, _, test_loader = load_fn(hdf5_path, config.batch_size)
    print(f"Data loaded successfully: {len(test_loader.dataset)} test samples")

    print("Building model and optimizer...")
    model, criterion, optimizer = build_fn(config.learning_rate, config.output_dir, device)
    print("Model and optimizer built successfully.")

    if not config.eval_only:
        print("Starting training...")
        train_loader, val_loader, _ = load_fn(hdf5_path, config.batch_size)
        early_stopping = EarlyStopping(patience=config.patience)
        model_checkpoint = ModelCheckpoint(output_dir=config.output_dir)
        
        train_fn(
            train_loader, 
            val_loader, 
            model, 
            criterion, 
            optimizer, 
            config.epochs, 
            config.patience, 
            config.output_dir, 
            mean_chm, 
            std_chm,
            device, 
            callbacks=[early_stopping, model_checkpoint]
        )
        print("Training completed.")

    print("Evaluating on test data...")
    eval_fn(test_loader, model, criterion, config.output_dir, mean_chm, std_chm, device)
    print("Test evaluation completed.")

    print("CHM Predictor Process Completed.")


if __name__ == "__main__":
    parser = configargparse.ArgParser(default_config_files=['./configs/config.ini'])

    parser.add('--config', is_config_file=True, help='Path to config file')
    parser.add('--data-folder', type=str, required=True, help='Root data directory containing Images and CHM folders')
    parser.add('--hdf5-file', type=str, default='Finland_CHM.h5', help='HDF5 file containing CHM dataset')
    parser.add('--output-dir', type=str, default='output/chmpredict', help='Directory to save output models and logs')
    parser.add('--learning-rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add('--batch-size', type=int, default=16, help='Batch size for DataLoader')
    parser.add('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add('--eval-only', action='store_true', help='Skip training and only perform evaluation')

    config, _ = parser.parse_known_args()

    main(config)


'''

Usage:

- Training

python -m chmpredict.main --data-folder /path/to/data --learning-rate 1e-4 --batch-size 16 --epochs 50 --patience 5 --output-dir output/chmpredict

Or

python -m chmpredict.main --config ./configs/config.ini

- Evaluation

python -m chmpredict.main --config ./configs/config.ini --eval-only

'''