import torch
import os
import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

class Dataload(Dataset):
    def __init__(self, path, transform):
        self.path = path
        self.transform = transform
        self.train_volumes = []
        self.train_labels = []
        self.train_volumes = sorted(glob.glob(os.path.join(path, "train-volume", '*.jpg')))
        self.train_labels = sorted(glob.glob(os.path.join(path, "train-labels", '*.jpg')))

    def __len__(self):
      return len(self.train_volumes)

    def __getitem__(self, index):
        img = Image.open(self.train_volumes[index]).convert('L')
        label = Image.open(self.train_labels[index]).convert('L')
        img = self.transform(img)
        label = self.transform(label)
        return img, label

class Dataload_Test(Dataset):
    def __init__(self, path, transform):
        self.transform = transform
        self.test_volumes = []
        self.test_volumes = sorted(glob.glob(os.path.join(path, "test-volume", '*.jpg')))

    def __len__(self):
      return len(self.test_volumes)

    def __getitem__(self, index):
        img = Image.open(self.test_volumes[index]).convert('L')
        img = self.transform(img)
        return img