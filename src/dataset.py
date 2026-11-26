// dataset 
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from typing import Tuple, Optional


class SyntheticVisionDataset(Dataset):
    """
    Synthetic dataset for testing pipelines, profiling memory,
    and verifying CPU/GPU training loops without external downloads.
    """
    def __init__(self, num_samples: int = 500, img_size: int = 224, num_classes: int = 10):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = torch.randn(3, self.img_size, self.img_size)
        label = int(torch.randint(0, self.num_classes, (1,)).item())
        return image, label


def get_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Standard production augmentation pipeline:
    - Train: Random Resized Crop, Horizontal Flip, Color Jitter, Normalization
    - Val: Resize, Center Crop, Normalization
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def create_dataloaders(
    data_dir: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 2,
    img_size: int = 64,
    num_classes: int = 10,
    use_synthetic: bool = False,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates Train and Validation DataLoaders.
    Gracefully falls back to Synthetic data if no dataset folder is specified.
    Supports sub-sampling for fast, cool CPU training.
    """
    train_tf, val_tf = get_transforms(img_size)

    if use_synthetic or not data_dir or not os.path.exists(data_dir):
        print(f"[Dataset] Using synthetic dataset for demonstration/profiling.")
        train_ds = SyntheticVisionDataset(num_samples=400, img_size=img_size, num_classes=num_classes)
        val_ds = SyntheticVisionDataset(num_samples=100, img_size=img_size, num_classes=num_classes)
    else:
        train_path = os.path.join(data_dir, "train")
        val_path = os.path.join(data_dir, "val")
        
        train_ds = datasets.ImageFolder(train_path, transform=train_tf)
        val_ds = datasets.ImageFolder(val_path, transform=val_tf)

        # Subsample for CPU efficiency to avoid thermal stress
        if max_train_samples and max_train_samples < len(train_ds):
            indices = torch.randperm(len(train_ds))[:max_train_samples].tolist()
            train_ds = torch.utils.data.Subset(train_ds, indices)
            print(f"[Dataset] Fast CPU mode: subsampled training set to {len(train_ds)} images.")
        else:
            print(f"[Dataset] Training set size: {len(train_ds)} images.")

        if max_val_samples and max_val_samples < len(val_ds):
            val_indices = torch.randperm(len(val_ds))[:max_val_samples].tolist()
            val_ds = torch.utils.data.Subset(val_ds, val_indices)
            print(f"[Dataset] Fast CPU mode: subsampled validation set to {len(val_ds)} images.")
        else:
            print(f"[Dataset] Validation set size: {len(val_ds)} images.")

    # Optimized DataLoader: pin_memory=True for fast host-to-device transfers if CUDA is present
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader
