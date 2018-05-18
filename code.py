from pytorch_CelebAToYearbook_cDCGAN import *


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

import torchvision.models
from torchvision.models.resnet import model_urls
from torch.utils.data.sampler import SubsetRandomSampler


model = torchvision.models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load('model_test(1).pth'))


data_transforms = {
    'train': transforms.Compose([
    #use for both training and testing the classifier
    #----gan preproc
    transforms.Scale(64),
        transforms.CenterCrop(64),
    #----classifier training preproc
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        #for evaluating gan outputs
    transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}


batch_size = 15


dset = FolderWithImages(root='./yearbook/', input_transform=data_transforms['train'], target_transform=data_transforms['train']) #new to get indices??
dset.image_filenames.sort()
shuf = True #new

num_train = len(dset.image_filenames)
indices = list(range(num_train))
split = int(np.floor(0.2 * num_train))


np.random.seed(511)
np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)


train_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, sampler=train_sampler) 

valid_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, sampler=valid_sampler)

train_iter = iter(train_loader)
valid_iter = iter(valid_loader)
x,_,l = train_iter.next()
xv,_,lv = valid_iter.next()
print(x.shape)
print(xv.shape)





model = model.cuda()
for param in model.parameters():
    param.requires_grad = True
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

criterion = nn.CrossEntropyLoss()

max_epoch = 30
bsize = 64

num_iter = 0
running_loss = 0.0
running_corrects = 0.0
running_v_corrects = 0.0
for epoch in range(0,max_epoch):
    #exp_lr_scheduler.step()
    train_iter = iter(train_loader)
    valid_iter = iter(valid_loader)
    num_iter = 0
    num_valid_iter = 0
    running_loss = 0.0
    running_corrects = 0.0
    running_v_corrects = 0.0
    while num_iter < len(train_loader):
        try:
            inputs,_,label_ids = train_iter.next()
            inputs_v,_,label_ids_v = valid_iter.next()
        except:
            valid_iter = iter(valid_loader)
            num_valid_iter = 0
            running_v_corrects = 0
        labels = y_gender_[label_ids]
        labelsv = y_gender_[label_ids_v]
        labels_onehot = torch.FloatTensor(bsize, num_classes)
        labels_onehot.zero_()
        try:
          labels_onehot.scatter_(1, labels.view(bsize,-1), 1)
        except:
          print('Not Scattered')
        inputs,labels = Variable(inputs.cuda()),Variable(labels.cuda())
        inputsv,labelsv = Variable(inputs_v.cuda()),Variable(labelsv.cuda())
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
        outputv = model(inputsv)
        _, predsv = torch.max(outputsv, 1)
        running_loss += loss.cpu().data #* inputs.size(0)
        running_corrects += torch.sum(preds.data == labels.data)
        running_v_corrects += torch.sum(predsv.data == labelsv.data)
        num_iter += 1
        num_valid_iter +=1
        if num_iter%150 == 0:
          print('num_iter: ' + str(num_iter) + ' loss: ' + str(running_loss/num_iter) + ' corrects: ' + str(running_corrects/(bsize*num_iter)) + 'vcorrects' + str(running_v_corrects/(bsize*num_iter)))
    epoch_loss = running_loss / len(train_loader)
    #epoch_acc = running_corrects.double() / len(train_loader)
    
    print('Epoch: ' + str(epoch) + ' ' + str(epoch_loss))
    filepath = 'model_res18_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), filepath)
    print('Saved Model to: ' + str(filepath))


