import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class HARDataset(Dataset):
    """Human Activity Recognition Dataset."""
    
    def __init__(self, root='./data', train=True):
        self.root = root
        self.train = train
        
        # Load data
        self.data, self.labels = self._load_data()
        self.targets = self.labels
        
        # Normalize using dataset-wide mean and std
        self._normalize()
        
        # Activity labels
        self.classes = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS',
                        'SITTING', 'STANDING', 'LAYING']

    def _load_data(self):
        """Load and preprocess HAR data."""
        if self.train:
            data_path = os.path.join(self.root, 'HAR', 'train', 'X_train.txt')
            label_path = os.path.join(self.root, 'HAR', 'train', 'y_train.txt')
        else:
            data_path = os.path.join(self.root, 'HAR', 'test', 'X_test.txt')
            label_path = os.path.join(self.root, 'HAR', 'test', 'y_test.txt')
        
        data = np.loadtxt(data_path)
        labels = np.loadtxt(label_path) - 1  # Convert to 0-based indexing
        
        # Reshape for 1D CNN: [batch_size, channels, sequence_length]
        data = data.reshape(-1, 1, data.shape[1])

        return torch.FloatTensor(data), torch.LongTensor(labels)
    
    def _normalize(self):
        """Normalize data to mean 0 and std 1 (per feature)."""
        # Calculate mean and std per feature
        mean = self.data.mean(dim=0, keepdim=True)
        std = self.data.std(dim=0, keepdim=True)
        # Add small epsilon to prevent division by zero
        eps = 1e-8
        self.data = (self.data - mean) / (std + eps)
        
        # Store normalization parameters
        self.mean = mean
        self.std = std

    def _augment(self, data):
        """Apply data augmentation for training."""
        if not self.train:
            return data
            
        # Random noise (safe augmentation that doesn't change sequence length)
        if np.random.random() < 0.3:
            noise = torch.randn_like(data) * 0.01
            data = data + noise
            
        # Random scaling (safe augmentation that doesn't change sequence length)
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            data = data * scale
            
        # Random masking (safe augmentation that doesn't change sequence length)
        if np.random.random() < 0.3:
            mask = torch.rand_like(data) > 0.1
            data = data * mask
            
        return data

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        
        if self.train:
            data = self._augment(data)
            
        return data, label
