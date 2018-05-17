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


model = torchvision.models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load('model_test(1).pth'))


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

batch_size = 64


dset = FolderWithImages(root='./yearbook/', input_transform=data_transforms['train'], target_transform=data_transforms['train']) #new to get indices??
#dset.image_filenames.sort()
shuf = True #new
train_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuf)


train_iter = iter(train_loader)
x,_,l = train_iter.next()
print(x.shape)



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
for epoch in range(14,max_epoch):
    #exp_lr_scheduler.step()
    train_iter = iter(train_loader)
    num_iter = 0
    running_loss = 0.0
    running_corrects = 0.0
    while num_iter < len(train_loader):
        try:
          inputs,_,label_ids = train_iter.next()
        except:
          continue
        labels = y_gender_[label_ids]
        labels_onehot = torch.FloatTensor(bsize, num_classes)
        labels_onehot.zero_()
        try:
          labels_onehot.scatter_(1, labels.view(bsize,-1), 1)
        except:
          print('Not Scattered')
        inputs,labels = Variable(inputs.cuda()),Variable(labels.cuda())
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
        running_loss += loss.cpu().data #* inputs.size(0)
        running_corrects += torch.sum(preds.data == labels.data)
        num_iter += 1
        if num_iter%150 == 0:
          print('num_iter: ' + str(num_iter) + ' loss: ' + str(running_loss/num_iter) + ' corrects: ' + str(running_corrects/(bsize*num_iter)))
    epoch_loss = running_loss / len(train_loader)
    #epoch_acc = running_corrects.double() / len(train_loader)
    
    print('Epoch: ' + str(epoch) + ' ' + str(epoch_loss))
    filepath = 'model_res18_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), filepath)
    print('Saved Model to: ' + str(filepath))


