"""Data loading and preprocessing utilities for X-ray analysis."""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

logger = logging.getLogger(__name__)


class SyntheticXrayDataset(Dataset):
    """Synthetic X-ray dataset for demonstration purposes.
    
    This dataset generates synthetic chest X-ray-like images for research
    and educational purposes only.
    """
    
    def __init__(
        self,
        size: int = 1000,
        image_size: Tuple[int, int] = (224, 224),
        num_classes: int = 2,
        transform: Optional[Any] = None,
        class_names: Optional[List[str]] = None
    ):
        """Initialize synthetic dataset.
        
        Args:
            size: Number of samples to generate
            image_size: Size of generated images
            num_classes: Number of classes
            transform: Optional transforms to apply
            class_names: Names for classes
        """
        self.size = size
        self.image_size = image_size
        self.num_classes = num_classes
        self.transform = transform
        self.class_names = class_names or ["Normal", "Pneumonia"]
        
        # Generate synthetic data
        self.data = self._generate_synthetic_data()
        
    def _generate_synthetic_data(self) -> List[Dict[str, Any]]:
        """Generate synthetic X-ray-like images."""
        data = []
        
        for i in range(self.size):
            # Generate random class
            label = np.random.randint(0, self.num_classes)
            
            # Create synthetic X-ray-like image
            image = self._create_synthetic_xray(label)
            
            data.append({
                "image": image,
                "label": label,
                "class_name": self.class_names[label],
                "patient_id": f"SYNTH_{i:06d}",
                "study_id": f"STUDY_{i:06d}"
            })
            
        return data
    
    def _create_synthetic_xray(self, label: int) -> np.ndarray:
        """Create a synthetic X-ray-like image."""
        # Base image with noise
        image = np.random.normal(0.5, 0.1, self.image_size)
        
        # Add some structure based on label
        if label == 1:  # Pneumonia
            # Add some cloudy regions
            for _ in range(np.random.randint(1, 4)):
                center_x = np.random.randint(50, self.image_size[0] - 50)
                center_y = np.random.randint(50, self.image_size[1] - 50)
                radius = np.random.randint(20, 60)
                
                y, x = np.ogrid[:self.image_size[0], :self.image_size[1]]
                mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                image[mask] += np.random.normal(0.2, 0.1)
        
        # Normalize to [0, 1]
        image = np.clip(image, 0, 1)
        
        return (image * 255).astype(np.uint8)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index."""
        item = self.data[idx]
        
        # Convert to PIL Image
        image = Image.fromarray(item["image"], mode='L')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return {
            "image": image,
            "label": torch.tensor(item["label"], dtype=torch.long),
            "class_name": item["class_name"],
            "patient_id": item["patient_id"],
            "study_id": item["study_id"]
        }


def get_transforms(
    image_size: int = 224,
    augmentation: bool = True,
    normalize: bool = True
) -> Dict[str, Any]:
    """Get data transforms for training and validation.
    
    Args:
        image_size: Target image size
        augmentation: Whether to apply augmentation
        normalize: Whether to normalize images
        
    Returns:
        Dictionary containing train and val transforms
    """
    # Base transforms
    base_transforms = [
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel
    ]
    
    # Training transforms with augmentation
    if augmentation:
        train_transforms = transforms.Compose([
            *base_transforms,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
        ])
    else:
        train_transforms = transforms.Compose([
            *base_transforms,
            transforms.ToTensor(),
        ])
    
    # Validation transforms (no augmentation)
    val_transforms = transforms.Compose([
        *base_transforms,
        transforms.ToTensor(),
    ])
    
    # Add normalization if requested
    if normalize:
        normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225]
        )
        train_transforms = transforms.Compose([train_transforms, normalize_transform])
        val_transforms = transforms.Compose([val_transforms, normalize_transform])
    
    return {
        "train": train_transforms,
        "val": val_transforms,
        "test": val_transforms
    }


def create_data_loaders(
    config: Any,
    synthetic: bool = True
) -> Dict[str, DataLoader]:
    """Create data loaders for training, validation, and testing.
    
    Args:
        config: Configuration object
        synthetic: Whether to use synthetic data
        
    Returns:
        Dictionary containing data loaders
    """
    transforms_dict = get_transforms(
        image_size=config.data.image_size,
        augmentation=config.data.augmentation,
        normalize=config.data.normalize
    )
    
    if synthetic:
        # Create synthetic datasets
        train_size = int(1000 * config.data.train_split)
        val_size = int(1000 * config.data.val_split)
        test_size = 1000 - train_size - val_size
        
        train_dataset = SyntheticXrayDataset(
            size=train_size,
            image_size=(config.data.image_size, config.data.image_size),
            transform=transforms_dict["train"]
        )
        
        val_dataset = SyntheticXrayDataset(
            size=val_size,
            image_size=(config.data.image_size, config.data.image_size),
            transform=transforms_dict["val"]
        )
        
        test_dataset = SyntheticXrayDataset(
            size=test_size,
            image_size=(config.data.image_size, config.data.image_size),
            transform=transforms_dict["test"]
        )
    else:
        # TODO: Implement real dataset loading
        raise NotImplementedError("Real dataset loading not implemented yet")
    
    # Create data loaders
    data_loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            pin_memory=True
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=True
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=True
        )
    }
    
    logger.info(f"Created data loaders: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    
    return data_loaders


def get_class_weights(dataset: Dataset) -> torch.Tensor:
    """Calculate class weights for imbalanced datasets.
    
    Args:
        dataset: Dataset to analyze
        
    Returns:
        Class weights tensor
    """
    labels = []
    for i in range(len(dataset)):
        item = dataset[i]
        if isinstance(item, dict):
            labels.append(item["label"].item())
        else:
            labels.append(item[1].item())
    
    labels = np.array(labels)
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    
    # Calculate weights inversely proportional to class frequency
    class_weights = total_samples / (len(class_counts) * class_counts)
    
    return torch.FloatTensor(class_weights)
