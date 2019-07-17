import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model_name = '3_layered'
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3,3), padding=1, bias=False)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.batchnorm1 = nn.BatchNorm2d(16)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,3), padding=1, bias=False)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.batchnorm2 = nn.BatchNorm2d(32)
        
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,3), padding=1, bias=False)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        self.batchnorm3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3,3), padding=1, bias=False)
        torch.nn.init.xavier_uniform_(self.conv4.weight)
        self.batchnorm4 = nn.BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(128, 128, kernel_size=(3,3), padding=1, bias=False)
        torch.nn.init.xavier_uniform_(self.conv5.weight)
        self.batchnorm5 = nn.BatchNorm2d(128)
        
        self.conv6 = nn.Conv2d(128, 128, kernel_size=(3,3), padding=1, bias=False)
        torch.nn.init.xavier_uniform_(self.conv6.weight)
        self.batchnorm6 = nn.BatchNorm2d(128)
        
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((2,8))
        
        self.fc = nn.Linear(128*2*8, 2)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        
    def forward(self, x):
        temp = F.relu(self.batchnorm1(self.conv1(x)))
        temp = self.maxpool1(temp)
        temp = F.relu(self.batchnorm2(self.conv2(temp)))
        temp = self.maxpool2(temp)
        temp = F.relu(self.batchnorm3(self.conv3(temp)))
        temp = F.relu(self.batchnorm4(self.conv4(temp)))
        temp = F.relu(self.batchnorm5(self.conv5(temp)))
        temp = F.relu(self.batchnorm6(self.conv6(temp)))
        temp = self.adaptive_avg_pool(temp)
        scores = self.fc(temp.view(-1,128*2*8))
        return scores