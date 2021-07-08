import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

import os
from torchvision.utils import save_image
import glob
import math 
from sklearn.model_selection import train_test_split
import gc
import time
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import elasticdeform
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from model import *
from dataloader import *
from tqdm import trange

path = "/content/drive/MyDrive/U-Net_DataSet"
batch_size = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor()])
print("Starting to load dataset")
train_dataset = Dataload("/content/drive/MyDrive/U-Net_Train_Deformed", transform)
train_set, val_set = train_test_split(train_dataset, test_size=0.2, random_state=42)
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers = 2)
val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers = 2)

test_dataset = Dataload_Test("/content/drive/MyDrive/U-Net_DataSet", transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = 2)
print("Dataset loading done")
epoch_train_losses = []              # Defining an empty list to store the epoch losses
epoch_val_losses = []             
accu_train_epoch = []                # Defining an empty list to store the accuracy per epoch
accu_val_epoch = []

model = UNetPP(DenseBlock, BasicDownBlock, BasicUpBlock)
print("Model is loaded")
def make_tensor(tensor):
      if torch.cuda.is_available():
        return torch.cuda.FloatTensor(tensor)
      else:
        return torch.FloatTensor(tensor)

def get_model_summary(model, input_tensor_shape):
    summary(model.to(device), input_tensor_shape)
get_model_summary(model, (1, 512, 512))

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy(input, target)
        smooth = 1e-5
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice             # We are giving more weightage to Dice Loss

batch_size = 1
epochs = 10
lr = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss = BCEDiceLoss()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr)


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def train(model, dataset, optimizer, criterion, device):

    train_loss_batch = []
    accu_train_batch = []
    model.train()
    for idx,(images, labels) in enumerate(dataset):
        images = images.to(device)
        labels = labels.to(device)

        #Forward Pass
        output = model(make_tensor(images))
        output = torch.clip(output, 0.0025, 0.9975)            # I am clipping the output because if it becomes 0 or 1 then there is a chance that loss function can explode
        labels = torch.round(labels)
        train_loss = criterion(output, labels)
        train_loss_batch.append(train_loss)
        output = torch.round(output)
        acc = iou_score(output, labels)
        accu_train_batch.append(acc)
        print(f"Batch: {idx + 1}   Train Loss: {train_loss:.5f}   Accuracy: {acc:.5f}")
        # Backward
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
    epoch_train_losses.append(sum(train_loss_batch)/360)
    accu_train_epoch.append(sum(accu_train_batch)/360)
    print(f"Train Epoch Loss: {(sum(train_loss_batch)/360):.5f}   Train Epoch Accuracy: {(sum(accu_train_batch)/360):.5f}")

def eval(model, dataset, criterion, device):

    val_loss_batch = []
    accu_val_batch = []
    model.eval()
    for idx,(images, labels) in enumerate(dataset):
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)
            #Forward Pass
            output = model(make_tensor(images))
            torch.clip(output, 0.0025, 0.9975)
            # Loss
            val_loss = criterion(output, labels)
            val_loss_batch.append(val_loss)
            output = torch.round(output)
            acc = iou_score(output, labels)
            accu_val_batch.append(acc)
    epoch_val_losses.append((sum(val_loss_batch))/90)
    accu_val_epoch.append((sum(accu_val_batch))/90)
    print(f"Val Epoch Loss: {((sum(val_loss_batch))/90):.5f}   Val Epoch Accuracy: {((sum(accu_val_batch))/90):.5f}")



def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

for epoch in trange(epochs):
    
    start_time = time.monotonic()

    print(f"Epoch: {epoch + 1}")
    train(model, train_dataloader, optimizer, loss, device)
    eval(model, val_dataloader, loss, device)
    # if best_valid_loss > epoch_val_losses[-1]:
    #     best_valid_loss = epoch_val_losses[-1]
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_val_losses[-1],
            }, '/content/drive/MyDrive/U-Net++.pt')
    end_time = time.monotonic()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()
    print("\n\n\n TIME TAKEN FOR THE EPOCH: {} mins and {} seconds".format(epoch_mins, epoch_secs))

print("OVERALL TRAINING COMPLETE")