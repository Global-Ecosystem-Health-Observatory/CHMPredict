import torch

from chmpredict.model.network.unet import UNet


def build_fn(learning_rate, device):
    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, criterion, optimizer