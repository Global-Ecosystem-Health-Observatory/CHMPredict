import os
import torch
import configargparse

from chmpredict.data.loader import load_fn
from chmpredict.model.build import build_fn
from chmpredict.model.train import train_fn
from chmpredict.model.eval import eval_fn


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = load_fn(config.data_folder, config.batch_size)

    model, criterion, optimizer = build_fn(config.learning_rate, device)

    train_fn(train_loader, val_loader, model, criterion, optimizer, config.epochs, config.patience, config.output_dir, device)

    eval_fn(test_loader, model, criterion, config.output_dir, device)


if __name__ == "__main__":
    parser = configargparse.ArgParser(default_config_files=['./configs/config.ini'])

    parser.add('--config', is_config_file=True, help='Path to config file')
    parser.add('--data-folder', type=str, required=True, help='Root data directory containing Images and CHM folders')
    parser.add('--output-dir', type=str, default='output/chmpredict', help='Directory to save output models and logs')
    parser.add('--learning-rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add('--batch-size', type=int, default=16, help='Batch size for DataLoader')
    parser.add('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add('--patience', type=int, default=5, help='Patience for early stopping')

    config, _ = parser.parse_known_args()

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    main(config)


'''

Usage:

python -m chmpredict.main --rgb_dir /path/to/rgb --chm_dir /path/to/chm --learning_rate 1e-4 --batch_size 16 --num_epochs 50 --patience 5 --model_path best_model.pth

Example:

python -m chmpredict.main --rgb_dir /Users/anisr/Documents/CHM_Images/Images --chm_dir /Users/anisr/Documents/CHM_Images/CHM --learning_rate 1e-4 --batch_size 16 --num_epochs 50 --patience 5 --model_path best_model.pth

python -m chmpredict.main --config ./configs/config.ini

'''