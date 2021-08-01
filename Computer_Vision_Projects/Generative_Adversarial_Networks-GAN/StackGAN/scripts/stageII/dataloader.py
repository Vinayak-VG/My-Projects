import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import PIL
import pickle
from torch.utils.data import Dataset
from glob import glob
import gc
from torch.utils.data import DataLoader
os.environ["TOKENIZERS_PARALLELISM"] = "false"

batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CUB_Dataset_2(Dataset):
    def __init__(self, mode = 'train', transform = True):
        self.mode = mode
        self.transforms = transform
        self.bbox = self.load_bbox()     
        self._init_dataset()
        self.filenames = self.load_filenames()
        if transform:
            self._init_transform()

    def _init_dataset(self):
        self.files = []
        self.text_files = []
        self.stage1_img_files = []
        dirs = sorted(os.listdir(os.path.join("/content/CUB_200_2011/images")))
        if self.mode == 'train': 
            for dir in range(len(dirs)):
                files = sorted(glob(os.path.join("/content/CUB_200_2011/images", dirs[dir], '*.jpg')))         
                self.files += files
                text_file = sorted(glob(os.path.join("/content/birds/text_c10", dirs[dir], '*.txt')))
                self.text_files += text_file
                stage_1_files = sorted(glob(os.path.join("/content/drive/MyDrive/StackGAN/StackGAN-1_Images", dirs[dir], '*.png')))
                self.stage1_img_files += stage_1_files

        else:
            print("No Such Dataset Mode")
            return None

    def load_bbox(self):
        bbox_path = os.path.join('/content/CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        filepath = os.path.join('/content/CUB_200_2011/images.txt')
        df_filenames = pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        return filename_bbox

    def get_img(self, img_path, bbox):
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        if bbox is not None:
            R = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - R)
            y2 = np.minimum(height, center_y + R)
            x1 = np.maximum(0, center_x - R)
            x2 = np.minimum(width, center_x + R)
            img = img.crop([x1, y1, x2, y2])
        load_size = int(256 * 76 / 64)
        img = img.resize((load_size, load_size), PIL.Image.BILINEAR)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def load_filenames(self):
        filepath = os.path.join('/content/birds/train/filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames1 = pickle.load(f)
        filepath = os.path.join('/content/birds/test/filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames2 = pickle.load(f)
        filenames = filenames1 + filenames2
        return filenames

    def _init_transform(self):
        self.transform1 = transforms.Compose([
            #transforms.RandomCrop((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform = transforms.Compose([
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def __getitem__(self, index):
        key = self.filenames[index]
        bbox = self.bbox[key]
        img = self.get_img(self.files[index], bbox)
        text = self.text_files[index]
        stageI_img = Image.open(self.stage1_img_files[index]).convert('RGB')
        stageI_img = self.transform1(stageI_img)
        return stageI_img, img, text

    def __len__(self):
        return len(self.stage1_img_files)
