import os
from typing import Tuple
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR10_CLASSES = (
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)

# Standard CIFAR-10 distribution moments
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def obtain_transforms():
    """Formulate stochastic data transformations alongside deterministic evaluation tensors."""
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
    return train_tf, test_tf


def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """Instantiate CIFAR-10 loaders across training and validation partitions."""
    os.makedirs(data_dir, exist_ok=True)
    train_tf, test_tf = obtain_transforms()

    cifar_img_dir = os.path.join(data_dir, "cifar10")
    if os.path.exists(os.path.join(cifar_img_dir, "train")):
        train_set = datasets.ImageFolder(
            root=os.path.join(cifar_img_dir, "train"), transform=train_tf
        )
        test_set = datasets.ImageFolder(
            root=os.path.join(cifar_img_dir, "test"), transform=test_tf
        )
    else:
        train_set = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=train_tf
        )
        test_set = datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=test_tf
        )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )
    return train_loader, test_loader


def get_calibration_loader(
    data_dir: str = "./data",
    subset_size: int = 500,
    batch_size: int = 32
) -> DataLoader:
    """Isolate a representative calibration slice from training tensors without stochastic jitter."""
    _, test_tf = obtain_transforms()
    cifar_img_dir = os.path.join(data_dir, "cifar10")
    if os.path.exists(os.path.join(cifar_img_dir, "train")):
        train_set_clean = datasets.ImageFolder(
            root=os.path.join(cifar_img_dir, "train"), transform=test_tf
        )
    else:
        train_set_clean = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=test_tf
        )
    indices = list(range(min(subset_size, len(train_set_clean))))
    calib_subset = Subset(train_set_clean, indices)
    return DataLoader(calib_subset, batch_size=batch_size, shuffle=False)
