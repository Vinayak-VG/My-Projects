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

class Conditioning_Augmentation_StageII(nn.Module):
    def __init__(self):
        super(Conditioning_Augmentation_StageII, self).__init__()

        self.fc1 = nn.Linear(768, 256)
        self.relu = nn.ReLU()

    def forward(self, x):
        
        x = x.to(device)
        #print(x.shape)
        y = self.relu(self.fc1(x))
        u0 = y[:, :128]
        logvar = y[:, 128:]
        sigma0 = torch.exp(logvar/2)
        epsilon = torch.randn((x.shape[0], 128)).to(device)
        out = u0 + sigma0*epsilon
        return out, u0, logvar


class DownSample2(nn.Module):
    def __init__(self):
        super(DownSample2, self).__init__()

        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias = False)
        
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias = False)
        self.batchnorm2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias = False)
        self.batchnorm3 = nn.BatchNorm2d(512)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = x.to(device)
        x = self.relu(self.conv1(x))
        x = self.relu(self.batchnorm2(self.conv2(x)))
        x = self.relu(self.batchnorm3(self.conv3(x)))

        return x

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias = False)
        self.batchnorm1 = nn.BatchNorm2d(512)

        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias = False)
        self.batchnorm2 = nn.BatchNorm2d(512)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = x.to(device)
        identity = x
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.batchnorm2(self.conv2(x))
        x = x + identity
        x = self.relu(x)

        return x

class UpSampling2(nn.Module):
    def __init__(self):
        super(UpSampling2, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias = False)
        self.batchnorm1 = nn.BatchNorm2d(256)

        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias = False)
        self.batchnorm2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias = False)
        self.batchnorm3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias = False)
        self.batchnorm4 = nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, bias = False)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = x.to(device)
        x = self.relu(self.batchnorm1(self.conv1(self.upsample(x))))
        x = self.relu(self.batchnorm2(self.conv2(self.upsample(x))))
        x = self.relu(self.batchnorm3(self.conv3(self.upsample(x))))
        x = self.relu(self.batchnorm4(self.conv4(self.upsample(x))))
        x = self.conv5(x)

        return x

class StageII_GAN_Gen(nn.Module):
    def __init__(self, downsample, resblock, upsample, condaug2):
        super(StageII_GAN_Gen, self).__init__()

        self.downsample = downsample()
        self.resblock = resblock()
        self.upsample = upsample()
        self.CA2 = condaug2()
        self.conv = nn.Conv2d(640, 512, kernel_size=3, stride=1, padding=1, bias = False)
        self.batchnorm = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()
    def forward(self, x, text):

        x = x.to(device)
        text = text.to(device)
        text, u0, logvar = self.CA2(text)
        text = text.unsqueeze(2)
        text = text.unsqueeze(3)
        text = text.repeat(1, 1, 16, 16)
        x = self.downsample(x)
        x = torch.cat((x, text), 1)
        x = self.relu(self.batchnorm(self.conv(x)))
        x = self.resblock(self.resblock(self.resblock(self.resblock(x))))
        x = self.upsample(x)
        x = self.tanh(x)

        return x, u0, logvar

class DownSample3(nn.Module):
    def __init__(self):
        super(DownSample3, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size = 4, stride = 2, padding = 1, bias = False)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.batchnorm2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.batchnorm3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.batchnorm4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(512, 1024, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.batchnorm5 = nn.BatchNorm2d(1024)

        self.conv6 = nn.Conv2d(1024, 2048, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.batchnorm6 = nn.BatchNorm2d(2048)

        self.conv7 = nn.Conv2d(2048, 1024, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.batchnorm7 = nn.BatchNorm2d(1024)

        self.conv8 = nn.Conv2d(1024, 512, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.batchnorm8 = nn.BatchNorm2d(512)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x):

        x = x.to(device)
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.batchnorm2(self.conv2(x)))
        x = self.leakyrelu(self.batchnorm3(self.conv3(x)))
        x = self.leakyrelu(self.batchnorm4(self.conv4(x)))
        x = self.leakyrelu(self.batchnorm5(self.conv5(x)))
        x = self.leakyrelu(self.batchnorm6(self.conv6(x)))
        x = self.leakyrelu(self.batchnorm7(self.conv7(x)))
        x = self.leakyrelu(self.batchnorm8(self.conv8(x)))

        return x

class StageII_GAN_Dis(nn.Module):
    def __init__(self, downsample):
        super(StageII_GAN_Dis, self).__init__()
        
        self.fc0 = nn.Linear(768, 128)
        self.downsample = downsample()
        self.conv1 = nn.Conv2d(640, 512, kernel_size=3, stride=1, padding = 1, bias = False)
        self.batchnorm1 = nn.BatchNorm2d(512)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(512, 1, kernel_size = 4, stride = 4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, text):
        
        x = x.to(device)
        text = text.to(device)
        x = self.downsample(x)  
        text = self.fc0(text)
        text = text.unsqueeze(2)
        text = text.unsqueeze(3)
        text = text.repeat(1, 1, 4, 4)
        x = torch.cat((x, text), 1)
        x = self.leakyrelu(self.batchnorm1(self.conv1(x))) 
        x = self.sigmoid(self.conv2(x))  
        x = x.squeeze(3)
        x = x.squeeze(2)

        return x
