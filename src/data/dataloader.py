"""
SkinCap Dataset Dataloader

Self-contained dataloader for FairCBM project.
Handles Fitzpatrick skin type labels for fairness-aware training.
"""

import os
import sys
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


def map_skin_tone_to_fitzpatrick(skin_tone):
    """Map skin tone values to Fitzpatrick scale (1-6)."""
    if pd.isnull(skin_tone):
        return np.nan
    if 0 <= skin_tone <= 7:
        return 1
    elif 8 <= skin_tone <= 16:
        return 2
    elif 17 <= skin_tone <= 25:
        return 3
    elif 26 <= skin_tone <= 30:
        return 4
    elif skin_tone > 30:
        return 5
    else:
        return np.nan


default_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


class SkinCapDataset(Dataset):
    """
    SkinCap Dataset for concept-based dermatological diagnosis.
    
    Returns 4-tuple: (image, concept_labels, binary_label, fitzpatrick)
    - image: RGB image tensor [3, 224, 224]
    - concept_labels: Binary concept presence [num_concepts]
    - binary_label: Malignant (1) vs. Benign (0)
    - fitzpatrick: Fitzpatrick skin type (1-6 scale)
    """
    
    def __init__(self, root_dir, split='train', label_type='concept', transform=None,
                 raw_csv=None, train_split=0.8, val_split=0.1, test_split=0.1, seed=42):
        """
        Initialize dataset.
        
        Args:
            root_dir: Root directory containing images (or pre-split CSVs)
            split: 'train', 'val', or 'test'
            label_type: 'binary' or 'concept'
            transform: Image transformations
            raw_csv: Optional path to raw CSV file (will be split automatically)
            train_split: Training set proportion (if using raw_csv)
            val_split: Validation set proportion (if using raw_csv)
            test_split: Test set proportion (if using raw_csv)
            seed: Random seed for splitting
        """
        self.root_dir = root_dir
        self.split = split
        self.label_type = label_type
        self.transform = transform if transform is not None else default_transform
        
        # Try two approaches: pre-split CSVs or raw CSV with splitting
        if raw_csv and os.path.exists(raw_csv):
            # Load raw CSV and split
            df_full = pd.read_csv(raw_csv)
            
            # Preprocess: Fill NaNs in malignant using three_partition_label
            if 'three_partition_label' in df_full.columns:
                malignant_map = {'malignant': 1, 'benign': 0, 'non-neoplastic': 0}
                df_full['malignant'] = df_full['malignant'].fillna(
                    df_full['three_partition_label'].map(malignant_map)
                )
            
            # Map skin tone to Fitzpatrick if needed
            if 'fitzpatrick_scale' not in df_full.columns and 'skin_tone' in df_full.columns:
                df_full['fitzpatrick_scale'] = df_full['skin_tone'].apply(map_skin_tone_to_fitzpatrick)
            
            # Remove 'do not consider' rows
            if 'Do not consider this image' in df_full.columns:
                df_full = df_full[df_full['Do not consider this image'] != 1].copy()
            
            # Remove rows where fitzpatrick_scale == -1 or NaN
            df_full = df_full[df_full['fitzpatrick_scale'] != -1].copy()
            df_full = df_full.dropna(subset=['fitzpatrick_scale', 'malignant']).copy()
            
            # Split data
            train_df, temp_df = train_test_split(df_full, train_size=train_split, random_state=seed, stratify=df_full['malignant'])
            val_size_adjusted = val_split / (val_split + test_split)
            val_df, test_df = train_test_split(temp_df, train_size=val_size_adjusted, random_state=seed, stratify=temp_df['malignant'])
            
            # Select appropriate split
            if split == 'train':
                self.df = train_df.reset_index(drop=True)
            elif split == 'val':
                self.df = val_df.reset_index(drop=True)
            elif split == 'test':
                self.df = test_df.reset_index(drop=True)
            else:
                raise ValueError(f"Unknown split: {split}")
            
            # Image directory (parent of CSV or specified root_dir)
            if os.path.isdir(os.path.join(os.path.dirname(raw_csv), 'skincap')):
                self.img_dir = os.path.join(os.path.dirname(raw_csv), 'skincap')
            else:
                self.img_dir = root_dir
        else:
            # Look for pre-split CSVs
            csv_file = os.path.join(root_dir, f'skincap_{split}.csv')
            if not os.path.exists(csv_file):
                raise FileNotFoundError(
                    f"CSV file not found: {csv_file}\n"
                    f"Either provide pre-split CSVs or use raw_csv parameter."
                )
            
            self.df = pd.read_csv(csv_file)
            self.img_dir = os.path.join(root_dir, 'skincap')
        
        # Fitzpatrick labels
        if 'fitzpatrick_scale' not in self.df.columns:
            raise ValueError("CSV must contain 'fitzpatrick_scale' column")
        self.fitzpatrick = self.df['fitzpatrick_scale'].values.astype('float32')
        
        # Binary labels (malignant vs benign)
        if 'malignant' not in self.df.columns:
            raise ValueError("CSV must contain 'malignant' column")
        
        # Concept columns (23 morphological features)
        self.lesion_cols = [
            'Papule', 'Plaque', 'Pustule', 'Bulla', 'Patch', 'Nodule', 'Ulcer',
            'Crust', 'Erosion', 'Atrophy', 'Exudate', 'Telangiectasia', 'Scale',
            'Scar', 'Friable', 'Warty/Papillomatous', 'Dome-shaped',
            'Brown(Hyperpigmentation)', 'White(Hypopigmentation)', 'Purple',
            'Yellow', 'Black', 'Erythema'
        ]
        
        # Verify all concept columns exist
        missing_cols = [col for col in self.lesion_cols if col not in self.df.columns]
        if missing_cols and label_type == 'concept':
            raise ValueError(f"Missing concept columns: {missing_cols}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get item by index.
        
        Returns:
            For label_type='binary': (image, binary_label, fitzpatrick)
            For label_type='concept': (image, concept_labels, binary_label, fitzpatrick)
        """
        row = self.df.iloc[idx]
        
        # Load image
        img_path = os.path.join(self.img_dir, row['skincap_file_path'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Fitzpatrick label
        fitzpatrick = torch.tensor(row['fitzpatrick_scale'], dtype=torch.float32)
        
        if self.label_type == 'binary':
            # Binary classification only
            label = torch.tensor(row['malignant'], dtype=torch.float32)
            return image, label, fitzpatrick
        
        elif self.label_type == 'concept':
            # Concept-based classification
            concept_labels = torch.tensor(row[self.lesion_cols].values.astype(np.float32))
            binary_label = torch.tensor(row['malignant'], dtype=torch.float32)
            return image, concept_labels, binary_label, fitzpatrick
        
        else:
            raise ValueError("label_type must be 'binary' or 'concept'")


def get_dataloader(root_dir, split='train', batch_size=32, shuffle=None, 
                   num_workers=4, label_type='concept', transform=None,
                   raw_csv=None, train_split=0.8, val_split=0.1, test_split=0.1, seed=42):
    """
    Create DataLoader for SkinCap dataset.
    
    Args:
        root_dir: Root directory with images
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        shuffle: Shuffle data (default: True for train, False otherwise)
        num_workers: Number of workers for data loading
        label_type: 'binary' or 'concept'
        transform: Image transformations
        raw_csv: Optional path to raw CSV (will be split automatically)
        train_split: Training proportion (if using raw_csv)
        val_split: Validation proportion (if using raw_csv)
        test_split: Test proportion (if using raw_csv)
        seed: Random seed for splitting
    
    Returns:
        DataLoader instance
    """
    if shuffle is None:
        shuffle = (split == 'train')
    
    dataset = SkinCapDataset(
        root_dir=root_dir,
        split=split,
        label_type=label_type,
        transform=transform,
        raw_csv=raw_csv,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        seed=seed
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


if __name__ == "__main__":
    # Test dataloader
    print("Testing SkinCap dataloader...")
    
    # Test with concept labels
    dataset = SkinCapDataset(
        root_dir='data/skincap',
        split='train',
        label_type='concept'
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of concepts: {len(dataset.lesion_cols)}")
    
    # Test loading one sample
    image, concepts, label, fitzpatrick = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Concepts shape: {concepts.shape}")
    print(f"Binary label: {label.item()}")
    print(f"Fitzpatrick type: {fitzpatrick.item()}")
    
    # Test dataloader
    loader = get_dataloader(
        root_dir='data/skincap',
        split='train',
        batch_size=4,
        label_type='concept'
    )
    
    for batch in loader:
        images, concepts, labels, fitzpatrick = batch
        print(f"\nBatch:")
        print(f"  Images: {images.shape}")
        print(f"  Concepts: {concepts.shape}")
        print(f"  Labels: {labels.shape}")
        print(f"  Fitzpatrick: {fitzpatrick.shape}")
        break
    
    print("\n✓ Dataloader test passed!")
