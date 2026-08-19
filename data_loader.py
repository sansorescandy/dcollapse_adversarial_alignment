import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_transforms(is_training=True):
    """
    Defines standard ResNet-50 transforms.
    Ensures input is 224x224 to prevent spatial degradation.
    """
    # ImageNet normalization statistics
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    if is_training:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

def get_dataloaders(data_dir, domain_name, batch_size=32, num_workers=4, is_source=True):
    """
    Loads an Office-31 (or Office-Home) domain (e.g., 'amazon', 'webcam', 'dslr').
    If is_source=True, splits into 80% train / 20% val safely preventing transform leakage.
    If is_source=False, loads the entire domain for target validation/evaluation.
    """
    domain_path = os.path.join(data_dir, domain_name)
    
    if not os.path.exists(domain_path):
        raise FileNotFoundError(f"Path not found: {domain_path}")

    if is_source:
        # Create two distinct dataset objects to prevent PyTorch Subset transform leakage
        full_dataset_train = datasets.ImageFolder(domain_path, transform=get_transforms(is_training=True))
        full_dataset_val = datasets.ImageFolder(domain_path, transform=get_transforms(is_training=False))
        
        # Calculate split sizes
        num_samples = len(full_dataset_train)
        train_size = int(0.8 * num_samples)
        
        # Generate deterministic random indices for the 80/20 split
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(num_samples, generator=generator).tolist()
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Assign subsets to their respective parent datasets with correct transforms
        train_dataset = Subset(full_dataset_train, train_indices)
        val_dataset = Subset(full_dataset_val, val_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        return train_loader, val_loader
        
    else:
        # For the target domain, everything is evaluation/validation (no labels used for training)
        target_dataset = datasets.ImageFolder(domain_path, transform=get_transforms(is_training=False))
        target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        
        return target_loader