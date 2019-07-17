#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Sampler

from PIL import Image

import torchvision.datasets as dset
import torchvision.transforms as T
import torchvision.models as models

import numpy as np
import os
import cv2

from datetime import datetime

from models import Net12, Net8, Net3


# In[ ]:


OUT_FILE = 'model_logs/general_log.txt'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dtype = torch.float32
print_every = 100

print('using device:', device)


# In[ ]:


# get image names that will be used
with open("line_img_names.txt","r") as file:
    line_file_names = file.read().split('\n')[:-1]

with open("hw_img_names.txt","r") as file:
    hw_file_names = file.read().split('\n')[:-1]


# In[ ]:


line_img_path = "/home/sadullah/line-type-classification/data/processed/line"
hw_img_path = "/home/sadullah/line-type-classification/data/processed/hwline"

line_data = []
hw_data = []

for line_file_name in line_file_names:
    line_data.append(cv2.imread(os.path.join(line_img_path, line_file_name)))
    
for hw_file_name in hw_file_names:
    hw_data.append(cv2.imread(os.path.join(hw_img_path, hw_file_name)))

# label of line is '0'
# label of hw is '1'

# sizes of validation data for line and handwritten sets
line_val_size = len(line_data) // 5
hw_val_size = len(hw_data) // 5

# sizes of test data for line and handwritten sets
line_test_size = len(line_data) // 5
hw_test_size = len(hw_data) // 5

# sizes of tran data for line and handwritten sets
line_train_size = len(line_data) - line_val_size - line_test_size
hw_train_size = len(hw_data) - hw_val_size - hw_test_size

# labels for line sets, which are zeros
train_label = np.zeros(line_train_size)
val_label = np.zeros(line_val_size)
test_label = np.zeros(line_test_size)

# add hw labels to the labels, hw labels are ones
train_label = np.concatenate((train_label,np.ones(hw_train_size)))
val_label = np.concatenate((val_label,np.ones(hw_val_size)))
test_label = np.concatenate((test_label,np.ones(hw_test_size)))

# split data into train, validation and test
train_data = line_data[ : line_train_size]
val_data = line_data[line_train_size : line_train_size + line_val_size]
test_data = line_data[line_train_size + line_val_size :]

# add hw data
train_data.extend(hw_data[ : hw_train_size])
val_data.extend(hw_data[hw_train_size : hw_train_size + hw_val_size])
test_data.extend(hw_data[hw_train_size + hw_val_size :])

# make necessary transposes for training model
train_data = np.array(train_data)
val_data = np.array(val_data)
test_data = np.array(test_data)

#data_means = np.mean(train_data, axis=(0,1,2))
#data_stds = np.std(train_data, axis=(0,1,2))


# In[ ]:


class AspectRatioSampler(Sampler):

    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size 
        self.n_batches = len(self.data_source.data) // self.batch_size + 1
        self.sorted = np.array(sorted(range(len(self.data_source.data)), key=lambda k : self.data_source.data[k].shape[1] / self.data_source.data[k].shape[0]))

    def __iter__(self):
        for i in range(self.n_batches):
            start_idx = np.random.randint(len(self.data_source.data) - self.batch_size)
            current_batch = self.sorted[start_idx:start_idx+self.batch_size]
            yield current_batch

    def __len__(self):
        return self.n_batches
        
class AlignCollate(object):
    
    def __init__(self):
        pass
    
    def __call__(self, batch):      
        images, labels = list(zip(*batch))
        
        images = list(images)
        labels = list(labels)
                
        images = self.resize(images)
        images = [T.ToTensor()(img) for img in images]
        
        images = torch.stack(images)
        labels = torch.LongTensor(np.array(labels))
        
        return images, labels

    def resize(self, batch):
        mean_width, mean_height = 0, 0
        for elem in batch:
            mean_height += elem.shape[0]
            mean_width += elem.shape[1]
        mean_width /= len(batch)
        mean_height /= len(batch)
        batch = [Image.fromarray(cv2.resize(elem, (round(mean_width), round(mean_height)))) for elem in batch]
        return batch


# In[ ]:


class HwPrintedDataset(Dataset):

    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        sample = self.data[i]
        return sample, self.label[i]
    
batch_size=7

train_dataset = HwPrintedDataset(train_data, train_label)
train_collate = AlignCollate()
train_loader = DataLoader(train_dataset, batch_sampler=AspectRatioSampler(train_dataset, batch_size), pin_memory=True,
                          collate_fn=train_collate)

val_dataset = HwPrintedDataset(val_data, val_label)
val_collate = AlignCollate()
val_loader = DataLoader(val_dataset, batch_sampler=AspectRatioSampler(val_dataset, batch_size), pin_memory=True,
                        collate_fn=val_collate)

test_dataset = HwPrintedDataset(test_data, test_label)
test_collate = AlignCollate()
test_loader = DataLoader(test_dataset, batch_sampler=AspectRatioSampler(test_dataset, batch_size), pin_memory=True,
                         collate_fn=test_collate)


# In[ ]:


def helper_accuracy_calculation(num_correct_0, num_correct_1, num_samples_0, num_samples_1):

    acc_0 = float(num_correct_0) / (num_samples_0 + 1e-9) * 100
    acc_1 = float(num_correct_1) / (num_samples_1 + 1e-9) * 100

    num_correct = num_correct_0 + num_correct_1
    num_samples = num_samples_0 + num_samples_1
    acc = float(num_correct) / num_samples * 100
    
    print('For line: %.2f' % acc_0)
    print('For hw: %.2f' % acc_1)

    print('General acc: %.2f' % acc)
    with open(OUT_FILE, 'a') as f:
        f.write('{:<15.4f}{:<15.4f}{:<15.4f}'.format(acc_0,acc_1,acc))
    
    return acc_0, acc_1, acc
    


# In[ ]:


def check_accuracy(val_loader, model):
    num_correct_0, num_correct_1, num_samples_0, num_samples_1 = 0, 0, 0, 0
    
    with torch.no_grad():
        for x, sample_batch in enumerate(val_loader):
            
            x, y = sample_batch
            
            x, y = x.to(device=device, dtype=dtype), y.to(device=device, dtype=torch.long)
            
            scores = model(x)
            _, preds = scores.max(1)
            
            num_correct_0 += torch.sum(preds[y == 0] == 0)
            num_samples_0 += torch.sum(y == 0)
            
            num_correct_1 += torch.sum(preds[y == 1] == 1) 
            num_samples_1 += torch.sum(y == 1)
        
        print('Checking accuracy on validation set')
        acc_0, acc1, acc = helper_accuracy_calculation(num_correct_0.item(), num_correct_1.item(), num_samples_0.item(), num_samples_1.item())
        
        return acc


# In[ ]:


def train_net(model, optimizer, train_loader, val_loader, epochs=5):
    with open(OUT_FILE, 'a') as f:
        f.write('\n\n\nTraining started at {} \n\n'.format(str(datetime.now())))
        f.write('Model name is {} \n\n'.format(model.model_name))
        f.write('{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}'.format('epoch','iteration', 'train_loss', 'train_hw_acc', 'train_line_acc', 'train_gen_acc', 'val_hw_acc', 'val_line_acc', 'val_gen_acc'))

    
    num_correct_0, num_correct_1, num_samples_0, num_samples_1 = 0, 0, 0, 0
    
    model = model.to(device=device)
    model.train()

    bets_model = None
    best_acc = 0

    for e in range(1,epochs+1):
        for t, sample_batch in enumerate(train_loader):
            
            x, y = sample_batch
            
            x, y = x.to(device=device, dtype=dtype), y.to(device=device, dtype=torch.long)
            
            scores = model(x)
            
            ### calclulation of training accuracy ###
            _, preds = scores.max(1)
            num_correct_0 += torch.sum(preds[y == 0] == 0)
            num_samples_0 += torch.sum(y == 0)

            num_correct_1 += torch.sum(preds[y == 1] == 1)
            num_samples_1 += torch.sum(y == 1)
            ### calclulation of training accuracy ###
            
            loss = F.cross_entropy(scores, y)
            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
            
            if t % print_every == 0:
                print('Epoch %s, Iteration %d, loss = %.4f' % (e, t, loss.item()))
                with open(OUT_FILE, 'a') as f:
                    f.write('\n{:<15}{:<15.4f}{:<15.4f}'.format(int(e),t,loss.item()))

                print('Checking accuracy on training set')
                acc_0, acc1, acc = helper_accuracy_calculation(num_correct_0.item(), num_correct_1.item(), num_samples_0.item(), num_samples_1.item())
                
                val_acc = check_accuracy(val_loader, model)
                if val_acc > best_acc:
                    best_model = model
                    best_acc = val_acc
                    time = datetime.now()
                    time = str(time.month) + "-" + str(time.day) + "-" + str(time.hour) + "-" + str(time.minute)
                    torch.save(model.state_dict(),'/home/sadullah/line-type-classification/data/scripts/model_logs/{0}_{1}_{2}_{3:.4f}_{4}.pth'.format(model.model_name,e,t,val_acc,time))
                print()
                
    with open(OUT_FILE, 'a') as f:
        f.write('\n\n\nTraining ended at {} \n\n\n'.format(str(datetime.now())))
        
        


# In[ ]:


learning_rate = 0.001
model = Net12()
#model.load_state_dict(torch.load('/home/sadullah/line-type-classification/data/scripts/model_logs/8_layered_3_1300_97.22347629796839.pth'))
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
train_net(model, optimizer, train_loader, val_loader)


# In[ ]:




