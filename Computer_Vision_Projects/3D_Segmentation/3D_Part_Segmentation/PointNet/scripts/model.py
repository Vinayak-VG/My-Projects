import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import math

batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

class T_Net(nn.Module):
    def __init__(self, out):
        super(T_Net, self).__init__()
        # In : (batch_size, n, 3)

        self.conv1 = nn.Conv1d(out, 64, kernel_size = 1, stride = 1)
        self.batchnorm1 = nn.BatchNorm1d(64)

        # (batch_size, n, 64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size = 1, stride = 1)
        self.batchnorm2 = nn.BatchNorm1d(128)

        # (batch_size, n, 128)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size = 1, stride = 1)
        self.batchnorm3 = nn.BatchNorm1d(1024)

        # (batch_size, 1024)
        self.fc1 = nn.Linear(1024, 512)
        self.batchnorm4 = nn.BatchNorm1d(512)
        # (batch_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.batchnorm5 = nn.BatchNorm1d(256)
        # (batch_size, 512)
        self.fc3 = nn.Linear(256, out*out) # out = 3 if input transform, else out = 128 if feature transform
        self.relu = nn.ReLU()
        

    def forward(self, x):
    
        x = x.to(device)
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.relu(self.batchnorm2(self.conv2(x)))
        x = self.relu(self.batchnorm3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = self.relu(self.batchnorm4(self.fc1(x)))
        x = self.relu(self.batchnorm5(self.fc2(x)))
        x = self.fc3(x)

        out = int(math.sqrt(x.shape[1]))
        iden = Variable(torch.from_numpy(np.eye(out).flatten().astype(np.float32))).view(1, out * out).repeat(x.shape[0], 1)
        iden = iden.to(device)
        x = x + iden
        x = torch.reshape(x,(-1, out, out))
        return x

class PointNet(nn.Module):
    def __init__(self, tnet):
        super(PointNet,self).__init__()
        # In : (batch_size, n, 3)
        self.tnet1 = tnet(3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.batchnorm1 = nn.BatchNorm1d(64)

        # (batch_size, n, 64)
        self.conv2 = nn.Conv1d(64, 128, 1) 
        self.batchnorm2 = nn.BatchNorm1d(128)

        # (batch_size, n, 128)
        self.conv3 = nn.Conv1d(128, 128, 1) 
        self.batchnorm3 = nn.BatchNorm1d(128)
        
        self.tnet2 = tnet(128)

        # (batch_size, n, 128)
        self.conv4 = nn.Conv1d(128, 512, 1)
        self.batchnorm4 = nn.BatchNorm1d(512)
        
        # (batch_size, n, 512)
        self.conv5 = nn.Conv1d(512, 2048, 1)
        self.batchnorm5 = nn.BatchNorm1d(2048)

        # (batch_size, 1088)        # We concatenate the global and local features
        self.conv6 = nn.Conv1d(4944 ,256, 1)
        self.batchnorm6 = nn.BatchNorm1d(256)

        # (batch_size, 512)        # We concatenate the global and local features
        self.conv7 = nn.Conv1d(256 ,256, 1)
        self.batchnorm7 = nn.BatchNorm1d(256)

        # (batch_size, 256)        # We concatenate the global and local features
        self.conv8 = nn.Conv1d(256 ,128, 1)
        self.batchnorm8 = nn.BatchNorm1d(128)

        # (batch_size, 128)
        self.conv9 = nn.Conv1d(128, 50, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x, label):

        x = x.to(device)
        out_tnet1 = self.tnet1(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, out_tnet1)
        x = x.transpose(2, 1)
        x = self.relu(self.batchnorm1(self.conv1(x)))
        out1 = x
        x = self.relu(self.batchnorm2(self.conv2(x)))
        out2 = x
        x = self.relu(self.batchnorm3(self.conv3(x)))
        out3 = x
        out_tnet2 = self.tnet2(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, out_tnet2)
        x = x.transpose(2, 1)
        x = self.relu(self.batchnorm4(self.conv4(x)))
        out4 = x
        x = self.batchnorm5(self.conv5(x))
        out5 = x
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 2048)
        x = torch.cat([x,label.squeeze(1)],1)
        x = x.view(-1, 2048+16, 1).repeat(1, 1, 2048)
        x = torch.cat((x, out1, out2, out3, out4, out5), 1)
        x = self.relu(self.batchnorm6(self.conv6(x)))
        x = self.relu(self.batchnorm7(self.conv7(x)))
        x = self.relu(self.batchnorm8(self.conv8(x)))
        x = self.conv9(x)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, 50), dim=-1)
        x = x.view(batch_size, 2048, 50) # [B, N, 50]

        return x, out_tnet2