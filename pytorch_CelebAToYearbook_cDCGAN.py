import os, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image

# G(z)
class generator(nn.Module):
    # initializers
    def __init__(self, d=128, cdim=2):
        super(generator, self).__init__()
        self.deconv1_1 = nn.ConvTranspose2d(100, d*4, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d*4)
        self.deconv1_2 = nn.ConvTranspose2d(cdim, d*4, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(d*4)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        # self.deconv4 = nn.ConvTranspose2d(d, 3, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    # def forward(self, input):
    def forward(self, input, label):
        x = F.leaky_relu(self.deconv1_1_bn(self.deconv1_1(input)), 0.2)
        y = F.leaky_relu(self.deconv1_2_bn(self.deconv1_2(label)), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.deconv2_bn(self.deconv2(x)), 0.2)
        x = F.leaky_relu(self.deconv3_bn(self.deconv3(x)), 0.2)
        # x = F.tanh(self.deconv4(x))
        x = F.leaky_relu(self.deconv4_bn(self.deconv4(x)), 0.2)
        x = F.tanh(self.deconv5(x))

        return x

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=128, cdim=2):
        super(discriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(3, d//2, 4, 2, 1)
        self.conv1_2 = nn.Conv2d(cdim, d//2, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        # self.conv4 = nn.Conv2d(d*4, 1, 4, 1, 0)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    # def forward(self, input):
    def forward(self, input, label):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        y = F.leaky_relu(self.conv1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        # x = F.sigmoid(self.conv4(x))
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


#image loading
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    # img = imageio.imread(filepath) #maybe this will be better?
    return img

class FolderWithImages(torch.utils.data.Dataset):
    def __init__(self, root, input_transform=None, target_transform=None):
        super(FolderWithImages, self).__init__()
        self.image_filenames = [os.path.join(root, x)
                                for x in os.listdir(root) if is_image_file(x.lower())]
        self.image_filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = input.copy()
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        # return input, target
        return input, target, index

    def __len__(self):
        return len(self.image_filenames)



# label preprocess
# with open('data/resized_celebA/gender_label.pkl', 'rb') as fp:
with open('./year_label_years6.pkl', 'rb') as fp:
# with open('data/resized_yearbook_decade/year_label_decade.pkl', 'rb') as fp:
# with open('data/resized_yearbook_years2/year_label_years2.pkl', 'rb') as fp:
# with open('data/resized_yearbook_years4/year_label_years4.pkl', 'rb') as fp:
    y_gender_ = pickle.load(fp)

y_gender_ = torch.LongTensor(y_gender_).squeeze()

num_classes = len(np.unique(y_gender_))
cont_class = False
img_size = 64
onehot = torch.zeros(num_classes, num_classes)
onehot = onehot.scatter_(1, torch.LongTensor(range(num_classes)).view(num_classes, 1), 1).view(num_classes, num_classes, 1, 1)
fill = torch.zeros([num_classes, num_classes, img_size, img_size])
for i in range(num_classes):
    fill[i, i, :, :] = 1

'''
# fixed noise & label
temp_z0_ = torch.randn(4, 100)
temp_z0_ = torch.cat([temp_z0_, temp_z0_], 0)
temp_z1_ = torch.randn(4, 100)
temp_z1_ = torch.cat([temp_z1_, temp_z1_], 0)

fixed_z_ = torch.cat([temp_z0_, temp_z1_], 0)
fixed_y_ = torch.cat([torch.zeros(4), torch.ones(4), torch.zeros(4), torch.ones(4)], 0).type(torch.LongTensor).squeeze()

fixed_z_ = fixed_z_.view(-1, 100, 1, 1)
fixed_y_label_ = onehot[fixed_y_]
fixed_z_, fixed_y_label_ = Variable(fixed_z_.cuda(), volatile=True), Variable(fixed_y_label_.cuda(), volatile=True)

def show_result(num_epoch, show = False, save = False, path = 'result.png'):
    G.eval()
    test_images = G(fixed_z_, fixed_y_label_)
    G.train()

    size_figure_grid = 4
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(size_figure_grid*size_figure_grid):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i, j].cla()
        ax[i, j].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
'''

# fixed noise & label
num_grid = 5
temp_z_ = torch.randn(num_grid, 100)
fixed_y_ = torch.zeros(num_grid, 1)
classesToShow = [1] #binary2
# classesToShow =  [1,4,6] #year small, decade
# classesToShow =  [1,2] #years2
# classesToShow =  [1] #years4
fixed_z_ = temp_z_


for i in classesToShow:
    temp = torch.zeros(num_grid, 1) + i
    fixed_y_ = torch.cat([fixed_y_, temp], 0)
    fixed_z_ = torch.cat([fixed_z_, temp_z_], 0)
# temp_z1_ = torch.randn(4, 100)
# temp_z1_ = torch.cat([temp_z1_, temp_z1_], 0)

# print(fixed_z_)
# print(fixed_y_)
# fixed_y_ = torch.cat([torch.zeros(num_grid), torch.ones(num_grid)], 0).type(torch.LongTensor).squeeze()
# temp_z0_ = torch.cat([temp_z0_]*(len(classesToShow)+1), 0)
fixed_z_ = fixed_z_.view(-1, 100, 1, 1)
fixed_y_ = fixed_y_.type(torch.LongTensor).squeeze()
fixed_y_label_ = onehot[fixed_y_]
#fixed_z_, fixed_y_label_ = Variable(fixed_z_.cuda(), volatile=True), Variable(fixed_y_label_.cuda(), volatile=True)

def show_result(num_epoch, show = False, save = False, path = 'result.png'):

    G.eval()
    test_images = G(fixed_z_, fixed_y_label_)
    G.train()

    #change these later...
    # num_grid = 4
    size_figure_grid = num_grid
    num_rows = len(classesToShow)+1
    # num_rows = 4
    fs = (int(num_rows*1.251),int(size_figure_grid*1.251))
    fig, ax = plt.subplots(num_rows, size_figure_grid, figsize=(num_grid+1, num_rows+1))
    for i, j in itertools.product(range(num_rows), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(num_rows*size_figure_grid):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i, j].cla()
        # ax[i, j].imshow(test_images[k, 0].cpu().data.numpy(), cmap='gray')
        ax[i, j].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
        # ax[i, j].imshow((test_images[k, 0].cpu().data.numpy()+1)/2, cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

# def show_noise_morp(show=False, save=False, path='result.png'):
#     source_z_ = torch.randn(10, 100)
#     z_ = torch.zeros(100, 100)
#     for i in range(5):
#         for j in range(10):
#             z_[i*20 + j] = (source_z_[i*2+1] - source_z_[i*2]) / 9 * (j+1) + source_z_[i*2]

#     for i in range(5):
#         z_[i*20+10:i*20+20] = z_[i*20:i*20+10]

#     #more interp here?
#     y_ = torch.cat([torch.zeros(10, 1), torch.ones(10, 1)], 0).type(torch.LongTensor).squeeze()
#     y_ = torch.cat([y_, y_, y_, y_, y_], 0)
#     y_label_ = onehot[y_]
#     y_label_ = y_label_.view(-1, num_classes, 1, 1)
#     z_ = z_.view(-1, 100, 1, 1)
    
#     z_, y_label_ = Variable(z_.cuda(), volatile=True), Variable(y_label_.cuda(), volatile=True)

#     G.eval()
#     test_images = G(z_, y_label_)
#     G.train()

#     size_figure_grid = 10
    
#     #new
#     test_images.data = test_images.data.mul(0.5).add(0.5)
#     vutils.save_image(test_images.data, path, nrow=size_figure_grid)

    # fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(img_size, img_size))
    # for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
    #     ax[i, j].get_xaxis().set_visible(False)
    #     ax[i, j].get_yaxis().set_visible(False)

    # for k in range(10 * 10):
    #     i = k // 10
    #     j = k % 10
    #     ax[i, j].cla()
    #     ax[i, j].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

    # if save:
    #     plt.savefig(path)

    # if show:
    #     plt.show()
    # else:
    #     plt.close()


'''
def show_noise_samples(show=False, save=False, path='result_samples.png'):
    # z_ = torch.randn(36, 100)
    size_figure_grid = 8
    z_ = torch.randn(size_figure_grid**2, 100)
    
    #gender
    # y_ = torch.cat([torch.zeros(18, 1), torch.ones(18, 1)], 0).type(torch.LongTensor).squeeze() #gender and year small
    #year full
    # y_ = torch.cat([torch.zeros(10, 1), torch.ones(10, 1)*91], 0).type(torch.LongTensor).squeeze()
    # y_ = torch.cat([torch.ones(10, 1)*64, torch.ones(10, 1)*91], 0).type(torch.LongTensor).squeeze() #year full
    # y_ = torch.cat([torch.ones(size_figure_grid**2//2, 1)*1, torch.ones(size_figure_grid**2//2, 1)*6], 0).type(torch.LongTensor).squeeze() #decade full
    # y_ = torch.cat([torch.ones(size_figure_grid**2//2, 1)*0, torch.ones(size_figure_grid**2//2, 1)*2], 0).type(torch.LongTensor).squeeze() #years2
    # todo: for binary2, generate noise according to case3 (5/6 old class 0, 1/6 new class 1)
    #todo: for years2, generate uniformly across all 3 labels
    # y_ = torch.cat([torch.ones(size_figure_grid**2//3, 1)*0, torch.ones(size_figure_grid**2//3, 1)*1, torch.ones(size_figure_grid**2//3 + 1, 1)*2], 0).type(torch.LongTensor).squeeze() #years2, all 3
    # y_ = torch.cat([torch.ones(size_figure_grid**2*5//6 + 1, 1)*0, torch.ones(size_figure_grid**2//6, 1)*1], 0).type(torch.LongTensor).squeeze() #binary2, unbalanced
    # y_ = torch.cat([torch.zeros(size_figure_grid**2//2, 1), torch.ones(size_figure_grid**2//2, 1)], 0).type(torch.LongTensor).squeeze() #binary2, balanced
    y_ = torch.cat([torch.ones(size_figure_grid**2//2, 1)*0, torch.ones(size_figure_grid**2//2, 1)*1], 0).type(torch.LongTensor).squeeze() #years4
    #another version of year? this may require changing other lines below
    # y_ = torch.ones(20, 1).type(torch.LongTensor).squeeze()
    # y_ = torch.cat([y_*0, y_*44, y_*64, y_*91, y_*30], 0)
    # y_ = torch.cat([torch.zeros(32, 1)+cont_ep, torch.ones(32, 1)-cont_ep], 0).type(torch.LongTensor).squeeze() #gender and year small
    # y_ = torch.cat([torch.ones(18, 1)*(1./3+cont_ep), torch.ones(18, 1)*(2./3+cont_ep)], 0).type(torch.LongTensor).squeeze() #gender and year small
    if cont_class:
        y_label_ = onehot[y_]
        y_label = y_label_.view(-1, 1, 1, 1)
    else:
        y_label_ = onehot[y_]
        y_label_ = y_label_.view(-1, num_classes, 1, 1)
    z_ = z_.view(-1, 100, 1, 1)
    
    z_, y_label_ = Variable(z_.cuda(), volatile=True), Variable(y_label_.cuda(), volatile=True)

    G.eval()
    test_images = G(z_, y_label_)
    G.train()

    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(img_size, img_size))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(size_figure_grid * size_figure_grid):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i, j].cla()
        ax[i, j].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
        # ax[i, j].imshow((test_images[k, 0].cpu().data.numpy()+1)/2, cmap='gray')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
'''

# def show_noise_samplesV(show=False, save=False, path='result_samples.png'):
#     # z_ = torch.randn(36, 100)
#     size_figure_grid = 8
#     z_ = torch.randn(size_figure_grid**2, 100)
    
#     #gender
#     # y_ = torch.cat([torch.zeros(18, 1), torch.ones(18, 1)], 0).type(torch.LongTensor).squeeze() #gender and year small
#     #year full
#     # y_ = torch.cat([torch.zeros(10, 1), torch.ones(10, 1)*91], 0).type(torch.LongTensor).squeeze()
#     # y_ = torch.cat([torch.ones(10, 1)*64, torch.ones(10, 1)*91], 0).type(torch.LongTensor).squeeze() #year full
#     # y_ = torch.cat([torch.ones(size_figure_grid**2//2, 1)*1, torch.ones(size_figure_grid**2//2, 1)*6], 0).type(torch.LongTensor).squeeze() #decade full
#     # y_ = torch.cat([torch.ones(size_figure_grid**2//2, 1)*0, torch.ones(size_figure_grid**2//2, 1)*2], 0).type(torch.LongTensor).squeeze() #years2
#     # todo: for binary2, generate noise according to case3 (5/6 old class 0, 1/6 new class 1)
#     #todo: for years2, generate uniformly across all 3 labels
#     # y_ = torch.cat([torch.ones(size_figure_grid**2//3, 1)*0, torch.ones(size_figure_grid**2//3, 1)*1, torch.ones(size_figure_grid**2//3 + 1, 1)*2], 0).type(torch.LongTensor).squeeze() #years2, all 3
#     y_ = torch.cat([torch.ones(size_figure_grid**2*5//6 + 1, 1)*0, torch.ones(size_figure_grid**2//6, 1)*1], 0).type(torch.LongTensor).squeeze() #binary2, unbalanced
#     # y_ = torch.cat([torch.zeros(size_figure_grid**2//2, 1), torch.ones(size_figure_grid**2//2, 1)], 0).type(torch.LongTensor).squeeze() #binary2, balanced
#     # y_ = torch.cat([torch.ones(size_figure_grid**2//2, 1)*0, torch.ones(size_figure_grid**2//2, 1)*1], 0).type(torch.LongTensor).squeeze() #years4
#     #another version of year? this may require changing other lines below
#     # y_ = torch.ones(20, 1).type(torch.LongTensor).squeeze()
#     # y_ = torch.cat([y_*0, y_*44, y_*64, y_*91, y_*30], 0)
#     # y_ = torch.cat([torch.zeros(32, 1)+cont_ep, torch.ones(32, 1)-cont_ep], 0).type(torch.LongTensor).squeeze() #gender and year small
#     # y_ = torch.cat([torch.ones(18, 1)*(1./3+cont_ep), torch.ones(18, 1)*(2./3+cont_ep)], 0).type(torch.LongTensor).squeeze() #gender and year small
#     if cont_class:
#         y_label_ = onehot[y_]
#         y_label = y_label_.view(-1, 1, 1, 1)
#     else:
#         y_label_ = onehot[y_]
#         y_label_ = y_label_.view(-1, num_classes, 1, 1)
#     z_ = z_.view(-1, 100, 1, 1)
    
#     z_, y_label_ = Variable(z_.cuda(), volatile=True), Variable(y_label_.cuda(), volatile=True)

#     G.eval()
#     test_images = G(z_, y_label_)
#     G.train()

#     #new
#     test_images.data = test_images.data.mul(0.5).add(0.5)
#     vutils.save_image(test_images.data,path)


# training parameters
#for celebA, still works decent on binary2
# batch_size = 128
# lr = 0.0002
# train_epoch = 20

# #next try for binary2, reduce batch size from 128 to 64
#changed both batch size and data loader, next double check by switching back to 128
# batch_size = 64
# lr = 0.0002
# train_epoch = 30

# #works! now just try decades with 128?
# batch_size = 128
# lr = 0.0002
# train_epoch = 30

#years2, batch_size 128 looks unstable so try 64, 
#also didnt work so lower step size from 0.0002 to 0.00005
#decrease further to 0.00002? still bad
#try disc_updates 5, stay at 64,0.00005, maybe better?
#next increase batch size to 128? keep 00005 and 5
#increase disc_updates from 5 to 8, keep 128 and.00005
# reduce batch_size now 8,64,.00005
#todo: either increase disc_updates or batch size
# batch_size = 64
# lr = 0.00005
# train_epoch = 30
# disc_updates = 8


# #REDO everything with sorted img list
# #binary2
# batch_size = 64
# lr = 0.0002
# train_epoch = 30
# disc_updates = 1

#decades
# batch_size = 128 #start wiht 128, maybe down to 64?
# lr = 0.0002
# train_epoch = 25
# disc_updates = 1
# #ok, maybe fine tune later?

#years2, try to balance the 3 classes when samp
# batch_size = 64
# lr = 0.00005
# train_epoch = 30
# disc_updates = 8
#does not work, what about 128? also increase disc_updates form 8 to 10
# batch_size = 128
# lr = 0.00005
# train_epoch = 25
# disc_updates = 10
#change the learning rate schedule? just for this, make every 15 its
# batch_size = 128
# lr = 0.00005
# train_epoch = 45
# disc_updates = 10

#just skip and try years4 with highly unbalanced??
#batch size 64 likely too small, but try it... 64,0.00005,25,1
#disc too powerful, increase disc_updates from 1 to 10
#todo: try increasing batch from 64 to 128
# batch_size = 64
# lr = 0.00005
# train_epoch = 25
# disc_updates = 10

#AGAIN rerun with x_idx
#binary2
# batch_size = 64
# lr = 0.0002
# train_epoch = 25
# disc_updates = 1

#now skip to years4 since thats what we might use?
#is there a bug? both 64 and 128
# batch_size = 64
# lr = 0.00005
# train_epoch = 25
# disc_updates = 10

#looks like years4 isnt working, 
#try years2 and/or decades for a sanity check?
# first try baseline params
# batch_size = 64
# lr = 0.0002
# train_epoch = 25
# disc_updates = 1
#maybe hold off on increasing disc_updates for now?
# batch_size = 64
# lr = 0.00005
# train_epoch = 25
# disc_updates = 1
#now try increasing disc_updates to 8
#didnt work, now try same things but with slower lr schedule
#todo: change lr schedule
# batch_size = 64
# lr = 0.00005
# train_epoch = 30
# disc_updates = 8

#now with new show_noise_samples function, try years4
#128 for any hope of training? out of memory
# batch_size = 64
# lr = 0.00005
# train_epoch = 25
# disc_updates = 10

#back to years4, do binary2 afterwards
# batch_size = 64
# lr = 0.00005
# train_epoch = 25
# disc_updates = 10

# #redo binary2 first
# batch_size = 64
# lr = 0.0002
# train_epoch = 25
# disc_updates = 1

#then maybe increase learning rate on years4 to 0.0002 and reduce disc_updates? check with notes
 # or go straight to years6?
batch_size = 64
lr = 0.0002
train_epoch = 25
disc_updates = 1

# data_loader
isCrop = False
if isCrop:
    transform = transforms.Compose([
        transforms.Scale(108),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
else:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
# data_dir = 'data/resized_celebA'          # this path depends on your computer
data_dir = 'data/resized_yearbook_binary2'          # this path depends on your computer
# data_dir = 'data/resized_yearbook_decade'          # this path depends on your computer
# data_dir = 'data/resized_yearbook_years2'          # this path depends on your computer
# data_dir = 'data/resized_yearbook_years4'          # this path depends on your computer
'''
dset = datasets.ImageFolder(data_dir, transform) #old
shuf = False
dset.imgs.sort()
train_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuf)
temp = plt.imread(train_loader.dataset.imgs[0][0])
if (temp.shape[0] != img_size) or (temp.shape[0] != img_size):
    sys.stderr.write('Error! image size is not 64 x 64! run \"yearbook_data_preprocess.py\" !!!')
    sys.exit(1)
'''
#dset = FolderWithImages(root=os.path.join(data_dir,'yearbook'), input_transform=transform, target_transform=transforms.ToTensor()) #new to get indices??
#sort the images??
#shuf = True #new
#train_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuf)
# '''


# # network
# #CHANGE THESE BATCH SIZES???
# G = generator(128, cdim=num_classes)
# D = discriminator(128, cdim=num_classes)
# # G = generator(batch_size, cdim=num_classes)
# # D = discriminator(batch_size, cdim=num_classes)
# G.weight_init(mean=0.0, std=0.02)
# D.weight_init(mean=0.0, std=0.02)
# G
# D

# # Binary Cross Entropy loss
# BCE_loss = nn.BCELoss()

# # Adam optimizer
# G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
# D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# # results save folder
# # root = 'CelebA_cDCGAN_results/'
# root = 'CelebAyearbook_binary2_cDCGAN_results/'
# model = 'CelebAyearbook_binary2_cDCGAN_year_'
# # root = 'CelebAyearbook_decade_cDCGAN_results/'
# # model = 'CelebAyearbook_decade_cDCGAN_year_'
# # root = 'CelebAyearbook_years2_cDCGAN_results/'
# # model = 'CelebAyearbook_years2_cDCGAN_year_'
# # root = 'CelebAyearbook_years4_cDCGAN_results/'
# # model = 'CelebAyearbook_years4_cDCGAN_year_'

# # model = 'CelebA_cDCGAN_'
# if not os.path.isdir(root):
#     os.mkdir(root)
# if not os.path.isdir(root + 'Fixed_results'):
#     os.mkdir(root + 'Fixed_results')

# train_hist = {}
# train_hist['D_losses'] = []
# train_hist['G_losses'] = []
# train_hist['per_epoch_ptimes'] = []
# train_hist['total_ptime'] = []

# print('training start!')
# start_time = time.time()
# for epoch in range(train_epoch):
#     D_losses = []
#     G_losses = []

#     # learning rate decay
#     '''
#     if (epoch+1) == 11:
#         G_optimizer.param_groups[0]['lr'] /= 10
#         D_optimizer.param_groups[0]['lr'] /= 10
#         print("learning rate change!")
#     if (epoch+1) == 16:
#         G_optimizer.param_groups[0]['lr'] /= 10
#         D_optimizer.param_groups[0]['lr'] /= 10
#         print("learning rate change!")
#     '''
#     if ((epoch+1) % 5 == 1) and (epoch > 8):
#     # if ((epoch+1) % 15 == 1) and (epoch > 8): #just for years2
#         G_optimizer.param_groups[0]['lr'] /= 10
#         D_optimizer.param_groups[0]['lr'] /= 10
#         print("learning rate change! now %f" % G_optimizer.param_groups[0]['lr'])


#     y_real_ = torch.ones(batch_size)
#     y_fake_ = torch.zeros(batch_size)
#     y_real_, y_fake_ = Variable(y_real_, Variable(y_fake_
#     epoch_start_time = time.time()
#     num_iter = 0
#     # for x_, _ in train_loader:
#     trn_iter = iter(train_loader) #new
#     while num_iter < len(train_loader):
#         #TODO: take a random permutation of both x_ and y_, so that classes are balanced within a batch
#         data = trn_iter.next()
#         x_,_,x_idx = data

#         if isCrop:
#             x_ = x_[:, :, 22:86, 22:86]

#         mini_batch = x_.size()[0]

#         if mini_batch != batch_size:
#             y_real_ = torch.ones(mini_batch)
#             y_fake_ = torch.zeros(mini_batch)
#             y_real_, y_fake_ = Variable(y_real_), Variable(y_fake_)
#             # y_ = y_gender_[batch_size*num_iter:]
#         else:
#             # y_ = y_gender_[batch_size*num_iter:batch_size*(num_iter+1)]
#             pass

#         #use same x on both generator and discriminataor?
#         if (num_iter % disc_updates == 0):
#             # train discriminator D
#             D.zero_grad()                      
            
#             #new
#             y_ = y_gender_[x_idx]
#             y_fill_ = fill[y_]
#             x_, y_fill_ = Variable(x_), Variable(y_fill_)

#             # print('D')
#             # print(x_.size(),x_idx.size(),y_.size(),y_fill_.size())
#             D_result = D(x_, y_fill_).squeeze()

#             D_real_loss = BCE_loss(D_result, y_real_)

#             z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
#             y_ = (torch.rand(mini_batch, 1) * num_classes).type(torch.LongTensor).squeeze()
#             y_label_ = onehot[y_]
#             y_fill_ = fill[y_]
#             z_, y_label_, y_fill_ = Variable(z_, Variable(y_label_, Variable(y_fill_)))

#             G_result = G(z_, y_label_)
#             D_result = D(G_result, y_fill_).squeeze()

#             D_fake_loss = BCE_loss(D_result, y_fake_)
#             D_fake_score = D_result.data.mean()
#             D_train_loss = D_real_loss + D_fake_loss

#             D_train_loss.backward()
#             D_optimizer.step()

#             D_losses.append(D_train_loss.data[0])

        
#         # train generator G
#         G.zero_grad()

#         z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
#         y_ = (torch.rand(mini_batch, 1) * num_classes).type(torch.LongTensor).squeeze()
#         y_label_ = onehot[y_]
#         y_fill_ = fill[y_]
#         z_, y_label_, y_fill_ = Variable(z_.cuda()), Variable(y_label_.cuda()), Variable(y_fill_.cuda())

#         G_result = G(z_, y_label_)
#         D_result = D(G_result, y_fill_).squeeze()

#         # print('G')
#         # print(z_.size(),y_.size(),y_fill_.size(),y_real_.size())
#         G_train_loss = BCE_loss(D_result, y_real_)

#         G_train_loss.backward()
#         G_optimizer.step()

#         G_losses.append(G_train_loss.data[0])

#         num_iter += 1

#         if (num_iter % 100) == 0:
#             print('%d - %d complete!' % ((epoch+1), num_iter))

#     epoch_end_time = time.time()
#     per_epoch_ptime = epoch_end_time - epoch_start_time

#     print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
#                                                               torch.mean(torch.FloatTensor(G_losses))))
#     fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
#     show_result((epoch+1), save=True, path=fixed_p)
#     train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
#     train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
#     train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

# end_time = time.time()
# total_ptime = end_time - start_time
# train_hist['total_ptime'].append(total_ptime)

# print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
# print("Training finish!... save training results")
# torch.save(G.state_dict(), root + model + 'generator_param.pkl')
# torch.save(D.state_dict(), root + model + 'discriminator_param.pkl')
# with open(root + model + 'train_hist.pkl', 'wb') as f:
#     pickle.dump(train_hist, f)

# show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')

# images = []
# for e in range(train_epoch):
#     img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
#     images.append(imageio.imread(img_name))
# imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)

# show_noise_morp(save=True, path=root + model + 'warp.png')
# # show_noise_samples(save=True, path=root + model + 'samp.png')
# show_noise_samplesV(save=True, path=root + model + 'samp.png')