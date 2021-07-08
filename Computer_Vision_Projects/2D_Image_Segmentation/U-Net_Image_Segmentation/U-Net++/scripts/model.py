import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DenseBlock(nn.Module):
    
    def __init__(self, inplanes, planes):
        super().__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(self.conv1.weight, mean=0.0, std=np.sqrt(2/(3*3*planes)))
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=np.sqrt(2/(3*3*planes)))
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(self.conv3.weight, mean=0.0, std=np.sqrt(2/(3*3*planes)))
        self.relu = nn.ReLU()
        self.gn = nn.GroupNorm(16, planes)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):

        out = self.conv1(x)
        out = self.gn(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.gn(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out

class BasicDownBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size = 3, stride = 1, padding = 1)
        nn.init.normal_(self.conv1.weight, mean=0.0, std=np.sqrt(2/(3*3*planes)))
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = 1, padding = 1)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=np.sqrt(2/(3*3*planes)))
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.relu = nn.ReLU()
        self.gn = nn.GroupNorm(16, planes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        out = self.conv1(x)
        out - self.gn(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn(out)
        out = self.relu(out)
        out1 = out
        out = self.maxpool(out)
        out = self.dropout(out)

        return out1, out

class BasicUpBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size = 3, stride = 1, padding = 1)
        nn.init.normal_(self.conv1.weight, mean=0.0, std=np.sqrt(2/(3*3*planes)))
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = 1, padding = 1)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=np.sqrt(2/(3*3*planes)))
        self.upconv = nn.ConvTranspose2d(in_channels = planes, out_channels = (planes//2), kernel_size = 2, stride = 2)
        nn.init.normal_(self.upconv.weight, mean=0.0, std=np.sqrt(2/(3*3*(planes//2))))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.upconv(out)
        out = self.dropout(out)

        return out

class UNetPP(nn.Module):
    def __init__(self, denseblock, basicdownblock, basicupblock):
        super(UNetPP, self).__init__()
        self.downlayer1 = basicdownblock(inplanes = 1, planes = 64)
        self.downlayer2 = basicdownblock(inplanes = 64, planes = 128)
        self.downlayer3 = basicdownblock(inplanes = 128, planes = 256)
        self.downlayer4 = basicdownblock(inplanes = 256, planes = 512)

        self.dense01 = denseblock(inplanes = 64*3, planes = 64)
        self.dense02 = denseblock(inplanes = 64*4, planes = 64)
        self.dense03 = denseblock(inplanes = 64*5, planes = 64)
        self.dense11 = denseblock(inplanes = 128*3, planes = 128)
        self.dense12 = denseblock(inplanes = 128*4, planes = 128)
        self.dense21 = denseblock(inplanes = 256*3, planes = 256)

        self.uplayer1 = basicupblock(inplanes = 512, planes = 1024)
        self.uplayer2 = basicupblock(inplanes = 1024, planes = 512)
        self.uplayer3 = basicupblock(inplanes = 256*3, planes = 256)
        self.uplayer4 = basicupblock(inplanes = 128*4, planes = 128)

        self.upconv1 = nn.ConvTranspose2d(in_channels = 128, out_channels = 128, kernel_size = 2, stride = 2)
        nn.init.normal_(self.upconv1.weight, mean=0.0, std=np.sqrt(2/(3*3*128)))
        self.upconv2 = nn.ConvTranspose2d(in_channels = 256, out_channels = 256, kernel_size = 2, stride = 2)
        nn.init.normal_(self.upconv2.weight, mean=0.0, std=np.sqrt(2/(3*3*256)))
        self.upconv3 = nn.ConvTranspose2d(in_channels = 512, out_channels = 512, kernel_size = 2, stride = 2)
        nn.init.normal_(self.upconv3.weight, mean=0.0, std=np.sqrt(2/(3*3*512)))

        #self.convds = nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size = 1, stride = 1)

        self.conv1 = nn.Conv2d(in_channels = 64*5, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        nn.init.normal_(self.conv1.weight, mean=0.0, std=np.sqrt(2/(3*3*64)))
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=np.sqrt(2/(3*3*64)))
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size = 1, stride = 1)
        nn.init.normal_(self.conv3.weight, mean=0.0, std=np.sqrt(2/(1*1*1)))
        self.relu = nn.ReLU()
        self.gn = nn.GroupNorm(16, 64)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):

        out00, x = self.downlayer1(x)
        out10, x = self.downlayer2(x)
        out10_u = self.upconv1(out10)
        out01 = torch.cat((out00, out10_u), 1)
        out01 = self.dense01(out01)
        #seg1 = self.sigmoid(self.convds(out01))
        out20, x = self.downlayer3(x)
        out20_u = self.upconv2(out20)
        out11 = torch.cat((out10, out20_u), 1)
        out11 = self.dense11(out11)
        out30, x = self.downlayer4(x)
        out30_u = self.upconv3(out30)
        out21 = torch.cat((out20, out30_u), 1)
        out21 = self.dense21(out21)
        out11_u = self.upconv1(out11)
        out02 = torch.cat((out00, out01, out11_u), 1)
        out02 = self.dense02(out02)
        #seg2 = self.sigmoid(self.convds(out02))
        out21_u = self.upconv2(out20)
        out12 = torch.cat((out10, out11, out21_u), 1)
        out12 = self.dense12(out12)
        out12_u = self.upconv1(out12)
        out03 = torch.cat((out00, out01, out02, out12_u), 1)
        out03 = self.dense03(out03)
        #seg3 = self.sigmoid(self.convds(out03))

        x = self.uplayer1(x)
        x = torch.cat((out30, x), 1)
        x = self.uplayer2(x)
        x = torch.cat((out20, out21, x), 1)
        x = self.uplayer3(x)
        x = torch.cat((out10, out11, out12, x), 1)
        x = self.uplayer4(x)
        x = torch.cat((out00, out01, out02, out03, x), 1)
        x = self.conv1(x)
        x = self.gn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.gn(x)
        x = self.relu(x)
        x = self.conv3(x)
        #seg4 = x
        seg4 = self.sigmoid(x)
        #seg = (seg1 + seg2 + seg3 + seg4)/4
        
        return seg4
