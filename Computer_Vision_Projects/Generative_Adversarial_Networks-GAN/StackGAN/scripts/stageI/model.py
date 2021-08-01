import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

class Conditioning_Augmentation_StageI(nn.Module):
    def __init__(self):
        super(Conditioning_Augmentation_StageI, self).__init__()

        self.fc1 = nn.Linear(768, 256)
        self.relu = nn.ReLU()

    def forward(self, x):
        
        x = x.to(device)
        y = self.relu(self.fc1(x))
        u0 = y[:, :128]
        logvar = y[:, 128:]
        sigma0 = torch.exp(logvar/2)
        epsilon = torch.randn((x.shape[0], 128)).to(device)
        out = u0 + sigma0*epsilon
        return out, u0, logvar

class StageI_GAN_Gen(nn.Module):
    def __init__(self, condaug1):
        super(StageI_GAN_Gen, self).__init__()

        # In: [batch_size, 128]
        self.CA1 = condaug1()

        self.fc = nn.Sequential(
            nn.Linear(228, 4*4*128*8),
            nn.BatchNorm1d(4*4*128*8),
            nn.ReLU(True))
        
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(128*8, 64*8, kernel_size=3, stride=1, padding=1, bias = False)
        self.batchnorm1 = nn.BatchNorm2d(64*8)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.Conv2d(64*8, 32*8, kernel_size=3, stride=1, padding=1, bias = False)
        self.batchnorm2 = nn.BatchNorm2d(32*8)

        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = nn.Conv2d(32*8, 16*8, kernel_size=3, stride=1, padding=1, bias = False)
        self.batchnorm3 = nn.BatchNorm2d(16*8)

        self.upsample4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4 = nn.Conv2d(16*8, 8*8, kernel_size=3, stride=1, padding=1, bias = False)
        self.batchnorm4 = nn.BatchNorm2d(8*8)

        self.conv5 = nn.Conv2d(8*8, 3, kernel_size=3, stride=1, padding=1, bias = False)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):

        x = x.to(device)
        x, u0, logvar = self.CA1(x)
        z = torch.randn((x.shape[0], 100)).to(device)
        x = torch.cat((x, z), 1)
        x = self.fc(x)
        x = torch.reshape(x, (-1, 128*8, 4, 4))
        x = self.relu(self.batchnorm1(self.conv1(self.upsample1(x))))
        x = self.relu(self.batchnorm2(self.conv2(self.upsample2(x))))
        x = self.relu(self.batchnorm3(self.conv3(self.upsample3(x))))
        x = self.relu(self.batchnorm4(self.conv4(self.upsample4(x))))
        x = self.tanh(self.conv5(x))

        return x, u0, logvar

class DownSample1(nn.Module):
    def __init__(self):
        super(DownSample1, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias = False)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias = False)
        self.batchnorm2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias = False)
        self.batchnorm3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias = False)
        self.batchnorm4 = nn.BatchNorm2d(512)

        self.leakyrelu = nn.LeakyReLU(0.2, inplace = True)

    def forward(self, x):
        
        x = x.to(device)
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.batchnorm2(self.conv2(x)))
        x = self.leakyrelu(self.batchnorm3(self.conv3(x)))
        x = self.leakyrelu(self.batchnorm4(self.conv4(x)))

        return x

class StageI_GAN_Dis(nn.Module):
    def __init__(self, downsample):
        super(StageI_GAN_Dis, self).__init__()

        self.fc1 = nn.Linear(768, 128)
        self.downsample = downsample()
        self.conv1 = nn.Conv2d(640, 512, kernel_size=1, stride=1, bias = False)
        self.batchnorm1 = nn.BatchNorm2d(512)
        self.leakyrelu = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(512, 1, kernel_size = 4, stride = 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, text):
        
        x = x.to(device)
        text = text.to(device)
        x = self.downsample(x)   
        text = self.fc1(text)
        text = text.unsqueeze(2)
        text = text.unsqueeze(3)
        text1 = torch.cat((text, text, text, text), 2)
        text = torch.cat((text1, text1, text1, text1), 3)
        x = torch.cat((x, text), 1)
        x = self.leakyrelu(self.batchnorm1(self.conv1(x))) 
        x = self.conv2(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)
        x = self.sigmoid(x)

        return x