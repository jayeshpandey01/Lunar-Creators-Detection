"""
Algorithm Overview for TIFF Image Dataset:
1. Data Loading:
   - Load TIFF images from M175124932LR directory
   - Handle train/val/test splits
   - Implement efficient caching and loading
2. Augmentation:
   - Apply specific augmentations for TIFF images
   - Maintain image quality
   - Handle spatial information
"""

import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from functools import lru_cache

class TiffDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_training=True, cache_size=1000):
        """
        Dataset for M175124932LR TIFF images
        Args:
            data_dir: Directory containing TIFF images (train/val/test split)
            transform: Albumentations transformations
            is_training: Whether this is for training
            cache_size: Number of images to cache
        """
        self.data_dir = data_dir
        self.transform = transform
        self.is_training = is_training
        self.cache_size = cache_size
        self.images = self._load_image_paths()
        
    @lru_cache(maxsize=1000)
    def _load_image(self, path):
        """Cached image loading with quality preservation"""
        try:
            # Try loading with PIL first for better TIFF support
            image = Image.open(path)
            image = np.array(image)
            if len(image.shape) == 2:  # Convert grayscale to RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 4:  # Convert RGBA to RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            return image
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return None
        
    def _load_image_paths(self):
        """Get paths for all TIFF images in the directory"""
        image_paths = []
        
        print(f"Loading images from: {self.data_dir}")
        
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Directory not found: {self.data_dir}")
            
        files = [f for f in os.listdir(self.data_dir) 
                if f.endswith(('.tiff', '.tif'))]
        
        print(f"Found {len(files)} TIFF images")
        files.sort()
        
        for file in files:
            image_paths.append(os.path.join(self.data_dir, file))
            
        if not image_paths:
            raise ValueError(f"No TIFF images found in {self.data_dir}")
            
        return image_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = self._load_image(img_path)
        
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Add channel dimension if needed
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        # Add batch dimension and ensure correct shape (B, C, H, W)
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Create a dummy target (0) since we don't have labels
        target = torch.tensor(0, dtype=torch.long)
        
        return image, target

def get_transforms(is_training=True, img_size=256):
    """
    Get image transformations optimized for TIFF images
    """
    if is_training:
        return A.Compose([
            A.RandomResizedCrop(img_size, img_size, scale=(0.8, 1.0)),
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.OneOf([
                A.ElasticTransform(
                    alpha=1, 
                    sigma=50, 
                    interpolation=1,
                    border_mode=4,
                    p=0.5
                ),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.5),
            ], p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

def create_data_loaders(config):
    """
    Create data loaders for train/val/test sets
    """
    train_dataset = TiffDataset(
        os.path.join(config['data']['train_data_dir'], 'M175124932LR'),
        transform=get_transforms(is_training=True, img_size=config['data']['img_size'])
    )
    
    val_dataset = TiffDataset(
        os.path.join(config['data']['val_data_dir'], 'M175124932LR'),
        transform=get_transforms(is_training=False, img_size=config['data']['img_size'])
    )
    
    test_dataset = TiffDataset(
        os.path.join(config['data']['test_data_dir'], 'M175124932LR'),
        transform=get_transforms(is_training=False, img_size=config['data']['img_size'])
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['dataloader']['batch_size'],
        shuffle=config['dataloader']['shuffle_train'],
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['dataloader']['batch_size'],
        shuffle=False,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['dataloader']['batch_size'],
        shuffle=False,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory']
    )
    
    return train_loader, val_loader, test_loader 