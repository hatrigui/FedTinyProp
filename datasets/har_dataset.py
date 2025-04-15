import os
import numpy as np
import torch
from torch.utils.data import Dataset

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
        
        # Pad to 576 features (24x24), reshape for CNN
        data = np.pad(data, ((0,0), (0,15)), 'constant')
        data = data.reshape(-1, 1, 24, 24)

        return torch.FloatTensor(data), torch.LongTensor(labels)
    
    def _normalize(self):
        """Normalize data to mean 0 and std 1 (per dataset)."""
        mean = self.data.mean()
        std = self.data.std()
        self.data = (self.data - mean) / std

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return data, label
