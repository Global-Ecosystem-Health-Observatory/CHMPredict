import os
import torch

from chmpredict.model.network.unet import UNet


def build_fn(learning_rate, output_dir, device):
    model = UNet(in_channels=3, out_channels=1).to(device)
    model = load_best_model(model, output_dir, device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, criterion, optimizer


def load_best_model(model, output_dir, device):
    best_model_path = os.path.join(output_dir, "best_model.pth")

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, weights_only=False, map_location=device))
        print(f"Best model loaded from {best_model_path}")
    else:
        print(f"No best model found at {best_model_path}. Please check the path.")
    return model