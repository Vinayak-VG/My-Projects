import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
from torchvision.utils import save_image
import glob
from torch.utils.data import Dataset
import elasticdeform
import numpy as np
from PIL import Image
from dataloader import Dataload

transforms = transforms.Compose([transforms.ToTensor(),])
train_dataset = Dataload("./inputs", transforms)
path = "./inputs/U-Net_Train_Deformed/"

for j in range(1,15):
    for i in range(30):
        X = train_dataset[i][0].numpy()              # Converting the tensor to numpy since we need to input a np array
        Y = train_dataset[i][1].numpy()              # Converting the tensor to numpy since we need to input a np array
        [X_deformed, Y_deformed] = elasticdeform.deform_random_grid([X, Y], sigma=10, points=3, axis = (1,2))        # The value of sigma and the dimension of the grid are given the U-Net paper
        fname1 = "train-volume/train-volume-" + str(i+1+j*30) + ".jpg"
        fname2 = "train-labels/train-labels-" + str(i+1+j*30) + ".jpg"
        save_image(torch.from_numpy(X_deformed.clip(0, 1).astype('float64')), os.path.join(path, fname1))     
        save_image(torch.from_numpy(Y_deformed.clip(0, 1).astype('float64')), os.path.join(path, fname2))