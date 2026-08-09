import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_transforms(is_training=True):
    """
    Define las transformaciones estándar para ResNet-50.
    Garantiza que la entrada sea de 224x224 para evitar la degradación espacial.
    """
    # Estadísticas de normalización de ImageNet
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
    Carga un dominio de Office-31 (ej. 'amazon', 'webcam', 'dslr').
    Si es el dominio fuente (is_source=True), divide en 80% train / 20% val.
    Si es el dominio destino (is_source=False), carga todo para validación/evaluación.
    """
    domain_path = os.path.join(data_dir, domain_name)
    
    if not os.path.exists(domain_path):
        raise FileNotFoundError(f"No se encontró la ruta: {domain_path}")

    if is_source:
        # Para el dominio fuente, necesitamos transformaciones de entrenamiento y validación
        full_dataset = datasets.ImageFolder(domain_path)
        
        # Split 80/20
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        # Semilla fija para reproducibilidad en los splits
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
        
        # Asignar transformaciones (un truco para aplicar distintas transformaciones a los splits)
        train_dataset.dataset.transform = get_transforms(is_training=True)
        val_dataset.dataset.transform = get_transforms(is_training=False)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        return train_loader, val_loader
        
    else:
        # Para el dominio destino, todo es evaluación/validación (no usamos etiquetas para entrenar)
        target_dataset = datasets.ImageFolder(domain_path, transform=get_transforms(is_training=False))
        target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        
        # Devolvemos el mismo loader dos veces para mantener compatibilidad de retornos si es necesario,
        # o simplemente un loader. Aquí retornamos solo uno para el target.
        return target_loader
