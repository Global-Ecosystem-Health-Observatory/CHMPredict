from torch.utils.data import DataLoader
from torchvision import transforms

from chmpredict.data.dataset import CHMDataset


def load_fn(hdf5_file_path, batch_size):
    data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),  # Ensure ToTensor() is the last in the RGB transform pipeline
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CHMDataset(hdf5_file_path, transform=data_transforms, mode="train")
    val_dataset = CHMDataset(hdf5_file_path, transform=data_transforms, mode="val")
    test_dataset = CHMDataset(hdf5_file_path, transform=data_transforms, mode="test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return train_loader, val_loader, test_loader