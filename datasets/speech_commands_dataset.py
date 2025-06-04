import os
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
import torch.nn.functional as F

class SpeechCommandsDataset(Dataset):
    """Speech Commands Dataset with MFCC preprocessing and optional augmentation."""

    def __init__(self, root='./data', split='train', mfcc_config=None, fixed_length: int = 32):
        self.root = root
        self.split = split  # 'train', 'val', or 'test'

        self.classes = [
            'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
            'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine',
            'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
            'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'
        ]
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.fixed_length = fixed_length
        # Default MFCC config
        self.mfcc_config = mfcc_config or {
            'sample_rate': 16000,
            'n_mfcc': 40,
            'n_fft': 1024,
            'hop_length': 512
        }

        self.base_dir = os.path.join(self.root, 'SpeechCommands', 'speech_commands_v0.02')
        self.file_list = self._load_file_list()
        self.data, self.labels = self._load_data()
        self.targets = self.labels
        self._normalize()

    def _load_file_list(self):
        """Load list of files for current split using official .txt files."""
        if self.split in ['val', 'test']:
            filename = 'validation_list.txt' if self.split == 'val' else 'testing_list.txt'
            list_path = os.path.join(self.base_dir, filename)
            with open(list_path, 'r') as f:
                file_rel_paths = [line.strip() for line in f.readlines()]
        elif self.split == 'train':
            # Exclude val/test files from the full list
            val_list = set()
            test_list = set()
            for name in ['validation_list.txt', 'testing_list.txt']:
                with open(os.path.join(self.base_dir, name)) as f:
                    if 'val' in name:
                        val_list = set(line.strip() for line in f)
                    else:
                        test_list = set(line.strip() for line in f)

            all_files = []
            for label in self.classes:
                class_path = os.path.join(self.base_dir, label)
                if not os.path.isdir(class_path):
                    continue
                for file in os.listdir(class_path):
                    rel_path = f"{label}/{file}"
                    if rel_path.endswith('.wav') and rel_path not in val_list and rel_path not in test_list:
                        all_files.append(rel_path)
            file_rel_paths = all_files
        else:
            raise ValueError(f"Unknown split: {self.split}")
        return file_rel_paths

    def _load_data(self):
        """Load MFCC-transformed audio tensors and labels."""
        data_list = []
        labels_list = []

        for rel_path in self.file_list:
            full_path = os.path.join(self.base_dir, rel_path)
            label = rel_path.split('/')[0]

            if label not in self.class_to_idx:
                continue  # skip unknown or background classes

            try:
                waveform, sample_rate = torchaudio.load(full_path)

                # Resample if needed
                if sample_rate != self.mfcc_config['sample_rate']:
                    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate,
                                                               new_freq=self.mfcc_config['sample_rate'])
                    waveform = resampler(waveform)

                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                # MFCC extraction
                mfcc_transform = torchaudio.transforms.MFCC(
                    sample_rate=self.mfcc_config['sample_rate'],
                    n_mfcc=self.mfcc_config['n_mfcc'],
                    melkwargs={
                        'n_fft': self.mfcc_config['n_fft'],
                        'hop_length': self.mfcc_config['hop_length']
                    }
                )
                mfcc = mfcc_transform(waveform).squeeze(0)  # [n_mfcc, time]

                if mfcc.ndim != 2:
                    continue

                data_list.append(mfcc)
                labels_list.append(self.class_to_idx[label])
            except Exception as e:
                print(f"[Warning] Skipped {rel_path}: {e}")
                continue

        if not data_list:
            raise RuntimeError("No valid audio samples found!")

        max_len = max(m.shape[1] for m in data_list)
        padded = [F.pad(m, (0, max_len - m.shape[1])) for m in data_list]
        data_tensor = torch.stack(padded)
        labels_tensor = torch.LongTensor(labels_list)
        return data_tensor, labels_tensor

    def _normalize(self):
        """Normalize across samples per feature (n_mfcc) and time."""
        mean = self.data.mean(dim=0, keepdim=True)
        std = self.data.std(dim=0, keepdim=True)
        eps = 1e-8
        self.data = (self.data - mean) / (std + eps)
        self.mean = mean
        self.std = std

    def _augment(self, x):
        if self.split != 'train':
            return x

        if np.random.rand() < 0.3:
            x += torch.randn_like(x) * 0.01

        if np.random.rand() < 0.3:
            stretch_factor = np.random.uniform(0.9, 1.1)
            new_len = int(x.shape[1] * stretch_factor)
            x = F.interpolate(x.unsqueeze(0), size=new_len, mode='linear', align_corners=False).squeeze(0)

        if np.random.rand() < 0.3:
            mask = (torch.rand_like(x) > 0.1).float()
            x *= mask

        return x

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        data = self._augment(data)

        # Force uniform length along time axis (dim=1)
        T = data.shape[1]
        if T < self.fixed_length:
            pad_amount = self.fixed_length - T
            data = F.pad(data, (0, pad_amount))
        elif T > self.fixed_length:
            data = data[:, :self.fixed_length]

        return data, label

