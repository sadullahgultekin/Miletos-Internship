import torch
import torch.nn as nn

batch_size = 64

class GeneratorNet(nn.Module):
    def __init__(self, batch_size):
        super(GeneratorNet, self).__init__()
        
        self.batch_size = batch_size
        
        self.fc1 = nn.Linear(96,1024)
        self.relu1 = nn.ReLU()
        self.batch1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024,7*7*128)
        self.relu2 = nn.ReLU()
        self.batch2 = nn.BatchNorm1d(7*7*128)
        self.convtrans1 = nn.ConvTranspose2d(128,64,4,stride=2,padding=1)
        self.relu3 = nn.ReLU()
        self.batch3 = nn.BatchNorm2d(64)
        self.convtrans2 = nn.ConvTranspose2d(64,1,4,stride=2,padding=1)
        self.tanh = nn.Tanh()
        
    def forward(self):
        x = 2 * torch.rand(batch_size, 96) - 1
        
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.batch1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.batch2(x)
        x = x.view(self.batch_size,128,7,7)
        x = self.convtrans1(x)
        x = self.relu3(x)
        x = self.batch3(x)
        x = self.convtrans2(x)
        x = self.tanh(x)
        x = x.view(-1)
        
        return x

class DiscriminatorNet(nn.Module):
    def __init__(self, batch_size):
        super(DiscriminatorNet, self).__init__()
        
        self.batch_size = batch_size
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1)
        self.leaky_relu1 = nn.LeakyReLU(0.01)
        self.max1 = nn.MaxPool2d(2,stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.leaky_relu2 = nn.LeakyReLU(0.01)
        self.max2 = nn.MaxPool2d(2,stride=2)
        self.fc1 = nn.Linear(4*4*64, 4*4*64)
        self.leaky_relu3 = nn.LeakyReLU(0.01)
        self.fc2 = nn.Linear(4*4*64, 1)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.leaky_relu1(x)
        x = self.max1(x)
        x = self.conv2(x)
        x = self.leaky_relu2(x)
        x = self.max2(x)
        x = x.view(self.batch_size,64*4*4)
        x = self.fc1(x)
        x = self.leaky_relu3(x)
        x = self.fc2(x)
        return x
   