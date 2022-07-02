# -*- coding: utf-8 -*-
"""
==========================
**Author**: Qian Wang, qian.wang173@hotmail.com
"""


from __future__ import print_function, division

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
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
from skimage import io, transform
import torch.nn.functional as F
import cv2
import skimage.measure
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True
import scipy
import scipy.io
import pdb
plt.ion()   # interactive mode
from sganet.model import CANNet
from sganet.model_mcnn import MCNN
from sganet.model_cffnet import CFFNet
from sganet.model_csrnet import CSRNet
from sganet.model_sanet import SANet
from sganet.model_tednet import TEDNet
#from cannet import CANNet
from sganet.myInception_segLoss import headCount_inceptionv3
from sganet.generate_density_map import generate_multi_density_map,generate_density_map


IMG_EXTENSIONS = ['.JPG','.JPEG','.jpg', '.jpeg', '.PNG', '.png', '.ppm', '.bmp', '.pgm', '.tif']
def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def make_dataset(dir, extensions):
    images = []
    dir = os.path.expanduser(dir)
    """
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
    """
    d = os.path.join(dir,'images')
    for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    image_path = os.path.join(root, fname)
                    head,tail = os.path.split(root)
                    label_path = os.path.join(head,'ground_truth','GT_'+fname[:-4]+'.mat')
                    item = [image_path, label_path]
                    images.append(item)

    return images

class ShanghaiTechDataset(Dataset):
    def __init__(self, data_dir, transform=None, phase='train',extensions=IMG_EXTENSIONS):
        self.samples = make_dataset(data_dir,extensions)
        self.image_dir = data_dir
        self.transform = transform
        self.phase = phase
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,idx):        
        img_file,label_file = self.samples[idx]
        image = cv2.imread(img_file)
        height, width, channel = image.shape
        annPoints = scipy.io.loadmat(label_file)
        annPoints = annPoints['image_info'][0][0][0][0][0]
        #randDrift = np.random.randint(-16,16,size=annPoints.shape)
        #if self.phase=='train' and np.random.random()>0.5:
        #    scale = np.random.random()*0.4+0.8
        #    image = cv2.resize(image,(np.int32(height*scale),np.int32(width*scale)))
        #    annPoints = np.int32(annPoints*scale)
        #positions = generate_density_map_adaptive_sigma(shape=image.shape,points=annPoints,sigma=4,k=5)
        positions = generate_density_map(shape=image.shape,points=annPoints,f_sz=15,sigma=4)
        fbs = generate_density_map(shape=image.shape,points=annPoints,f_sz=25,sigma=1)
        fbs = np.int32(fbs>0)
        shortEdge = np.minimum(height,width)
        #tmp = np.random.randint(0.2*shortEdge,0.5*shortEdge)
        tsize = [128,128]
        #tsize = positions.shape
        if self.phase=='train':
            targetSize = tsize
        else:
            targetSize = tsize
        height, width, channel = image.shape
        if height < tsize[0] or width < tsize[1]:
            image = cv2.resize(image,(np.maximum(tsize[0]+2,height),np.maximum(tsize[1]+2,width)))
            count = positions.sum()
            max_value = positions.max()
            # down density map
            positions = cv2.resize(positions, (np.maximum(tsize[0]+2,height),np.maximum(tsize[1]+2,width)))
            count2 = positions.sum()
            positions = np.minimum(positions*count/(count2+1e-8),max_value*10)
            fbs = cv2.resize(fbs,(np.maximum(tsize[0]+2,height),np.maximum(tsize[1]+2,width)))
            fbs = np.int32(fbs>0)
        #if self.phase == 'val' or self.phase == 'test':
        #    image = cv2.resize(image,None,fx=0.5,fy=0.5)
        # label = np.expand_dims(label, 2)
        # transpose from h x w x c to c x h x w
        # label = label.transpose(2,0,1)
        if len(image.shape)==2:
            image = np.expand_dims(image,2)
            image = np.concatenate((image,image,image),axis=2)
        # transpose from h x w x channel to channel x h x w
        image = image.transpose(2,0,1)
        
        numPatches = 4
        if self.phase == 'train':
            patchSet, countSet, fbsSet = getRandomPatchesFromImage(image,positions,fbs,targetSize,numPatches)
            x = np.zeros((patchSet.shape[0],3,targetSize[0],targetSize[1]))
            if self.transform:
              for i in range(patchSet.shape[0]):
                #transpose to original:h x w x channel
                x[i,:,:,:] = self.transform(np.uint8(patchSet[i,:,:,:]).transpose(1,2,0))
            patchSet = x
        if self.phase == 'val' or self.phase == 'test':
            # patchSet, countSet = getAllPatchesFromImage(image,positions,targetSize),positions
            
            patchSet, countSet, fbsSet = getAllFromImage(image, positions, fbs)
            patchSet[0,:,:,:] = self.transform(np.uint8(patchSet[0,:,:,:]).transpose(1,2,0))
        return patchSet, countSet, fbsSet

def getRandomPatchesFromImage(image,positions,fbs,target_size,numPatches):
    # generate random cropped patches with pre-defined size, e.g., 224x224
    imageShape = image.shape
    if np.random.random()>0.5:
        for channel in range(3):
            image[channel,:,:] = np.fliplr(image[channel,:,:])
        positions = np.fliplr(positions)
        fbs = np.fliplr(fbs)
    #if np.random.random()>0.8:
    #    for channel in range(3):
    #        image[channel,:,:] = np.flipud(image[channel,:,:])
    #    positions = np.flipud(positions)
    patchSet = np.zeros((numPatches,3,target_size[0],target_size[1]))
    # generate density map
    countSet = np.zeros((numPatches,1,target_size[0],target_size[1]))
    fbsSet = np.zeros((numPatches,1,target_size[0],target_size[1]))
    for i in range(numPatches):
        topLeftX = np.random.randint(imageShape[1]-target_size[0]+1)#x-height
        #if imageShape[2]-target_size[1]-1 < 1:
        #    pdb.set_trace()
        topLeftY = np.random.randint(imageShape[2]-target_size[1]+1)#y-width
        thisPatch = image[:,topLeftX:topLeftX+target_size[0],topLeftY:topLeftY+target_size[1]]
        # pdb.set_trace()
        patchSet[i,:,:,:] = thisPatch
        # density map
        position = positions[topLeftX:topLeftX+target_size[0],topLeftY:topLeftY+target_size[1]]
        fb = fbs[topLeftX:topLeftX+target_size[0],topLeftY:topLeftY+target_size[1]]
        #position = skimage.measure.block_reduce(position,(2,2),np.sum)
        '''
        count = position.sum()
        max_value = position.max()
        # down density map
        position = cv2.resize(position, (ht_1, wd_1))
        count2 = position.sum()
        position = np.minimum(position*count/(count2+1e-8),max_value*10)
        #position = position * ((112 * 112) / (ht_1 * wd_1))
        '''
        position = position.reshape((1, position.shape[0], position.shape[1]))
        fb = fb.reshape((1, fb.shape[0], fb.shape[1]))
        countSet[i,:,:,:] = position
        fbsSet[i,:,:,:] = fb
    return patchSet, countSet, fbsSet

def getAllPatchesFromImage(image,positions,target_size):
    # generate all patches from an image for prediction
    nchannel,height,width = image.shape
    nRow = np.int(height/target_size[1])
    nCol = np.int(width/target_size[0])
    target_size[1] = np.int(height/nRow)
    target_size[0] = np.int(width/nCol)
    patchSet = np.zeros((nRow*nCol,3,target_size[1],target_size[0]))
    # ht_1 = int(target_size[0] / 2)
    # wd_1 = int(target_size[1] / 2)
    # generate density map
    # countSet = np.ones((nRow*nCol,1,ht_1,wd_1))
    for i in range(nRow):
      for j in range(nCol):
        # pdb.set_trace()
        patchSet[i*nCol+j,:,:,:] = image[:,i*target_size[1]:(i+1)*target_size[1], j*target_size[0]:(j+1)*target_size[0]]
        #position = positions[i*target_size[1]:(i+1)*target_size[1], j*target_size[0]:(j+1)*target_size[0]]
        #position = skimage.measure.block_reduce(position,(2,2),np.sum)
        #position = position.reshape((1, position.shape[0], position.shape[1]))
        #countSet[i*nCol+j,:,:,:] = position
    return patchSet#, countSet

def getAllFromImage(image,positions,fbs):
    nchannel, height, width = image.shape
    patchSet =np.zeros((1,3,height, width))
    patchSet[0,:,:,:] = image[:,:,:]
    countSet = positions.reshape((1,1,positions.shape[0], positions.shape[1]))
    fbsSet = fbs.reshape((1,1,fbs.shape[0], fbs.shape[1]))
    return patchSet, countSet, fbsSet

data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        #transforms.RandomResizedCrop((256)),
        #transforms.Resize((128,128)),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        #transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.ToPILImage(),
        #transforms.RandomResizedCrop((256)),
        #transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
class SubsetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

data_dir = './data/shanghaitech/part_B_final/'
#data_dir = '../DenseNetTest/data/original/shanghaitech/part_A_final/'
image_datasets = {x: ShanghaiTechDataset(data_dir+x+'_data', 
                        phase=x, 
                        transform=data_transforms[x])
                    for x in ['train','test']}
image_datasets['val'] = ShanghaiTechDataset(data_dir+'train_data',phase='val',transform=data_transforms['val'])
## split the data into train/validation/test subsets
indices = list(range(len(image_datasets['train'])))
split = np.int(len(image_datasets['train'])*0.2)
batch_size = 8

val_idx = np.random.choice(indices, size=split, replace=False)
train_idx = indices#list(set(indices)-set(val_idx))
test_idx = range(len(image_datasets['test']))

train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)
test_sampler = SubsetSampler(test_idx)

num_workers=4
train_loader = torch.utils.data.DataLoader(dataset=image_datasets['train'],batch_size=batch_size,sampler=train_sampler, num_workers=num_workers)
val_loader = torch.utils.data.DataLoader(dataset=image_datasets['val'],batch_size=1,sampler=val_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(dataset=image_datasets['test'],batch_size=1,sampler=test_sampler, num_workers=num_workers)

dataset_sizes = {'train':len(train_idx),'val':len(val_idx),'test':len(image_datasets['test'])}
dataloaders = {'train':train_loader,'val':val_loader,'test':test_loader}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
######################################################################
# Mixup data for data augmentation
# ^^^^^^^^^^^^^^^^^^^^^^
# https://github.com/hongyi-zhang/mixup/blob/master/cifar/utils.py
def mixup_data(x,y,use_cuda=True):
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    
    mixed_x = 0.5*x + 0.5*x[index,:]
    mixed_y = (y + y[index,:])>0
    mixed_y = mixed_y.float()
    return torch.cat((x,mixed_x),0), torch.cat((y,mixed_y),0)

def mixup_data2(x,y,use_cuda=True):
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    
    mixed_x = 0.5*x + 0.5*x[index,:]
    mixed_y = y
    return torch.cat((x,mixed_x),0), torch.cat((y,y),0)

######################################################################
# Visualize a few images
# ^^^^^^^^^^^^^^^^^^^^^^
# Let's visualize a few training images so as to understand the data
# augmentations.

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# Get a batch of training data
# Make a grid from batch
# out = torchvision.utils.make_grid(inputs)
######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_mae_val = 1e6
    best_mae_by_val = 1e6
    best_mae_by_test = 1e6
    best_mse_by_val = 1e6
    best_mse_by_test = 1e6
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
       
        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            
            # Iterate over data.
            for index, (inputs, labels, fbs) in enumerate(dataloaders[phase]):
                labels = labels*100
                labels1 = labels
                labels2 = skimage.measure.block_reduce(labels.cpu().numpy(),(1,1,1,2,2),np.sum)
                labels2 = torch.from_numpy(labels2)
                labels3 = skimage.measure.block_reduce(labels.cpu().numpy(),(1,1,1,4,4),np.sum)
                fbs3 = skimage.measure.block_reduce(fbs.cpu().numpy(),(1,1,1,4,4),np.max)
                fbs3 = np.float32(fbs3>0)
                #labels3 = labels1
                labels3 = torch.from_numpy(labels3)
                fbs3 = torch.from_numpy(fbs3)
                #labels4 = skimage.measure.block_reduce(labels.cpu().numpy(),(1,1,1,8,8),np.sum)
                #labels4 = torch.from_numpy(labels4)
                labels1 = labels1.to(device)
                labels2 = labels2.to(device)
                labels3 = labels3.to(device)
                fbs3 = fbs3.to(device)
                #labels4 = labels4.to(device)
                inputs = inputs.to(device)
                inputs = inputs.view(-1,inputs.shape[2],inputs.shape[3],inputs.shape[4])
                labels1 = labels1.view(-1,labels1.shape[3],labels1.shape[4])
                labels2 = labels2.view(-1,labels2.shape[3],labels2.shape[4])
                labels3 = labels3.view(-1,labels3.shape[3],labels3.shape[4])
                fbs3 = fbs3.view(-1,fbs3.shape[3],fbs3.shape[4])
                #labels4 = labels4.view(-1,labels4.shape[3],labels4.shape[4])
                # mixed up
                inputs = inputs.float()
                labels1 = labels1.float()
                labels2 = labels2.float()
                labels3 = labels3.float()
                #labels4 = labels4.float()
                #if epoch <= 20:
                #    inputs,labels = mixup_data(inputs,labels,use_cuda=True)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    output3,fbs_out = model(inputs)
                    criterion2 = nn.BCELoss()
                    #loss1 = criterion(output1, labels1)
                    #loss2 = criterion(output2, labels2)
                    #criterion = nn.L1Loss()
                    loss3 = criterion(output3, labels3)
                    loss_seg = criterion2(fbs_out, fbs3)
                    #th = 10*(epoch//100+1)
                    #th = 0.1*epoch+5 #cl2
                    th=1000
                    weights = th/(F.relu(labels3-th)+th)
                    loss3 = loss3*weights
                    loss3 = loss3.sum()/weights.sum()
                    #loss4 = criterion(output4, labels4)
                    #loss = (5*loss2 + loss3)/6.
                    loss = loss3 + 20*loss_seg
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                
            epoch_loss = running_loss / dataset_sizes[phase]
            
            
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            print()
        if epoch%1==0:
            tmp,epoch_mae,epoch_mse,epoch_mre=test_model(model,optimizer,'val')
            tmp,epoch_mae_test,epoch_mse_test,epoch_mre_test = test_model(model,optimizer,'test')
            if  epoch_mae < best_mae_val:
                best_mae_val = epoch_mae
                best_mae_by_val = epoch_mae_test
                best_mse_by_val = epoch_mse_test
                best_epoch_val = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
            if epoch_mae_test < best_mae_by_test:
                best_mae_by_test = epoch_mae_test
                best_mse_by_test = epoch_mse_test
                best_epoch_test = epoch
            print()
            print('best MAE and MSE by val: {} and {} at Epoch {}'.format(best_mae_by_val,best_mse_by_val, best_epoch_val))
            print('best MAE and MSE by test: {} and {} at Epoch {}'.format(best_mae_by_test,best_mse_by_test, best_epoch_test))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test_model(model,optimizer,phase):
    since = time.time()
    model.eval()
    mae = 0
    mse = 0
    mre = 0
    pred = np.zeros((3000,2))
    # Iterate over data.
    for index, (inputs, labels, fbs) in enumerate(dataloaders[phase]):
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = inputs.float()
        labels = labels.float()
        inputs = inputs.view(-1,inputs.shape[2],inputs.shape[3],inputs.shape[4])
        labels = labels.view(-1,labels.shape[3],labels.shape[4])
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs3,fbs_out = model(inputs)
            #outputs1 = outputs1.to(torch.device("cpu")).numpy()/100
            #outputs2 = outputs2.to(torch.device("cpu")).numpy()/100
            outputs3 = outputs3.to(torch.device("cpu")).numpy()/100
            #outputs4 = outputs4.to(torch.device("cpu")).numpy()/100
            pred_count = outputs3.sum()
        #pdb.set_trace()
        #utils.save_density_map(outputs, './', 'output_' + str(index) + '.png')
        true_count = labels.to(torch.device("cpu")).numpy().sum()
        # backward + optimize only if in training phase
        mse = mse + np.square(pred_count-true_count)
        mae = mae + np.abs(pred_count-true_count)
        mre = mre + np.abs(pred_count-true_count)/true_count
        pred[index,0] = pred_count
        pred[index,1] = true_count
    pred = pred[0:index+1,:]
    mse = np.sqrt(mse/(index+1))
    mae = mae/(index+1)
    mre = mre/(index+1)
    print(phase+':')
    print(mae,mse,mre)
    time_elapsed = time.time() - since
    #print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    #scipy.io.savemat('./results.mat', mdict={'pred': pred, 'mse': mse, 'mae': mae,'mre': mre})
    return pred,mae,mse,mre
    # load best model weights
    # return cmap,emap,p,r,f,outputs_test.to(torch.device("cpu")).numpy()


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrained model and reset final fully connected layer.
#
model = headCount_inceptionv3(pretrained=True)
#model = MCNN()
#model = SANet()
# model = TEDNet(use_bn=True)
# model = models.inception_v3(pretrained=True)
# model.fc = nn.Linear(2048,1)
model = model.to(device)

criterion = nn.MSELoss(reduce=False)

# Observe that all parameters are being optimized
# optimizer = optim.SGD(model.parameters(), lr=1e-7, momentum=0.95, weight_decay= 0)
#optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.95, weight_decay= 5e-4)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# optimizer = optim.RMSprop(model.parameters(), lr=1e-4)# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

#####################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 15-25 min on CPU. On GPU though, it takes less than a
# minute.
#
model_dir = './'
#model.load_state_dict(torch.load(model_dir+'sheepNet_model.pt'))
#test_model(model,optimizer,'test')
model = train_model(model, criterion, optimizer, exp_lr_scheduler,num_epochs=501)
pred,mae,mse,mre = test_model(model,optimizer,'test')
scipy.io.savemat('./results_inv3.mat', mdict={'pred': pred, 'mse': mse, 'mae': mae,'mre': mre})
torch.save(model.state_dict(), model_dir+'headCount_inv3.pt')

