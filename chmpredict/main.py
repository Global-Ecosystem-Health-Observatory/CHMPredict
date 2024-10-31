import os
import torch
import configargparse

from chmpredict.data.loader import load_fn
from chmpredict.data.create import create_patch_pool

from chmpredict.model.build import build_fn
from chmpredict.model.train import train_fn
from chmpredict.model.eval import eval_fn


def main(config):
    print("Starting CHM Predictor Training Process...")

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    hdf5_path = os.path.join(config.data_folder, config.hdf5_file)

    if not os.path.exists(hdf5_path):
        print(f"Creating patch pool at {hdf5_path}...")
        create_patch_pool(config.data_folder, config.hdf5_file, patch_size=256, stride=256)
    else:
        print(f"HDF5 file {hdf5_path} already exists. Skipping creation.")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    train_loader, val_loader, test_loader = load_fn(hdf5_path, config.batch_size)
    print(f"Data loaded successfully: {len(train_loader.dataset)} training samples, "
          f"{len(val_loader.dataset)} validation samples, {len(test_loader.dataset)} test samples")

    print("Building model and optimizer...")
    model, criterion, optimizer = build_fn(config.learning_rate, device)
    print("Model and optimizer built successfully.")

    print("Starting training...")
    train_fn(train_loader, val_loader, model, criterion, optimizer, config.epochs, config.patience, config.output_dir, device)
    print("Training completed.")

    print("Evaluating on test data...")
    test_loss = eval_fn(test_loader, model, criterion, config.output_dir, device)
    print(f"Test evaluation completed. Final test loss: {test_loss:.4f}")

    print("CHM Predictor Training Process Completed.")


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

    config, _ = parser.parse_known_args()

    main(config)


'''

Usage:

python -m chmpredict.main --data-folder /path/to/data --learning-rate 1e-4 --batch-size 16 --epochs 50 --patience 5 --output-dir output/chmpredict

Or

python -m chmpredict.main --config /path/to/config.ini



'''