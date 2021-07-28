import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import json
import glob
import math 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from model import *
from dataloader import *

TRAIN_DATASET = PartNormalDataset(npoints=2048, split='trainval', normal_channel=False)
TEST_DATASET = PartNormalDataset(npoints=2048, split='test', normal_channel=False)
trainloader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
testloader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

model = PointNet(T_Net).to(device)
model = model.apply(weights_init)

for idx,(data, labels, seg) in enumerate(testloader):
    
    data = data.to(device)
    labels = labels.to(device)
    seg = seg.to(device)
    print(data.shape)
    data = data.float()
    labels = labels.long()
    seg = seg.long()
    data = data.transpose(2, 1)

    output, A = model(data, to_categorical(labels, 16))
    data = data.transpose(2, 1)
    point_cloud = data
    k = 5     # You can change this number to test the results on different samples of test data
    xyz=data[k,:,:]
    xyz = torch.reshape(xyz, (xyz.shape[1], xyz.shape[0]))
    xyz = xyz.cpu().detach().numpy()
    _, out_seg = torch.max(output, 2)
    out = out_seg[k, :]
    rgb = np.zeros(shape=[2048, 3])
    partid = [[0, 1, 2, 3], [4, 5], [6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23], [24, 25, 26, 27], [28, 29], [30, 31, 32, 33, 34, 35], [36, 37], [38, 39, 40], [41, 42, 43], [44, 45, 46], [47, 48, 49]]
    unique = torch.unique(out).tolist()
    for i in range(2048):
        if out[i] == unique[0]:
            rgb[i,:] = [1, 0, 0]
        elif out[i] == unique[1]:
            rgb[i,:] = [0, 1, 0]
        elif out[i] == unique[2]:
            rgb[i,:] = [0, 0, 1]
        elif out[i] == unique[3]:
            rgb[i,:] = [1, 0.5, 1]
        elif out[i] == unique[4]:
            rgb[i,:] = [0, 1, 1]
        elif out[i] == unique[5]:
            rgb[i,:] = [1, 1, 0]
        else:
            rgb[i,:] = [0.5, 0.5, 0.5]
    ax = plt.axes(projection='3d')
    xyz = xyz.reshape((2048, 3))
    ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c = rgb)
    plt.grid(False)
    plt.axis('off')
    plt.show()
    break