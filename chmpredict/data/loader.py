import os

from torchvision import transforms
from torch.utils.data import DataLoader

from chmpredict.data.dataset import CHMDataset


def load_fn(data_folder, batch_size):
    rgb_dir = os.path.join(data_folder, "Images")
    chm_dir = os.path.join(data_folder, "CHM")

    data_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
        ]
    )

    train_dataset = CHMDataset(rgb_dir, chm_dir, transform=data_transforms, mode="train")
    val_dataset = CHMDataset(rgb_dir, chm_dir, mode="val")
    test_dataset = CHMDataset(rgb_dir, chm_dir, mode="test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader