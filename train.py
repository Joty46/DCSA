import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import time
import albumentations as A
from albumentations.pytorch import ToTensor
from torch.utils.data import random_split
from torch.optim import lr_scheduler
import seaborn as sns
import pandas as pd
import argparse
import os
from data_loading import multi_classes, binary_class  # Assuming you have your custom data loading module
from sklearn.model_selection import GroupKFold
from self_attention_cv import transunet
from pytorch_dcsaunet import DCSAU_Net
from loss import *


import cv2
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.ops as ops

import torch.nn as nn
import torch.nn.functional as F


# class OutlineAttention(nn.Module):
#     def __init__(self, in_channels):
#         super(OutlineAttention, self).__init__()
#         self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)
#         self.sigmoid = nn.Sigmoid()

#         # Sobel filters for gradient computation
#         self.sobel_x = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
#         self.sobel_y = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)

#         # Initialize Sobel filters
#         self.sobel_x.weight.data = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).view(1, in_channels, 3, 3)
#         self.sobel_y.weight.data = torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).view(1, in_channels, 3, 3)

#     def forward(self, x):
#         # Apply Sobel filtering
#         sobel_x = self.sobel_x(x)
#         sobel_y = self.sobel_y(x)
#         sobel = torch.sqrt(sobel_x.pow(2) + sobel_y.pow(2))

#         # Concatenate Sobel filtering result with the original input
#         x_sobel = torch.cat((x, sobel), dim=1)

#         # Apply attention mechanism
#         attention_map = self.sigmoid(self.conv(x_sobel))
#         attended_features = x * attention_map
#         return attended_features




def get_train_transform():
    return A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.25),
        A.ShiftScaleRotate(shift_limit=0, p=0.25),
        A.CoarseDropout(),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
    ])

def get_valid_transform():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
    ])

# class DCSAU_NetWithOutline(nn.Module):
#     def __init__(self, img_channels, n_classes):
#         super(DCSAU_NetWithOutline, self).__init__()
#         self.dcsaunet = DCSAU_Net.Model(img_channels=img_channels, n_classes=n_classes)
#         self.outline_attention = OutlineAttention(in_channels=1)  # Adjust in_channels here

#     def forward(self, x):
#         features = self.dcsaunet(x)
#         attended_features = self.outline_attention(features)
#         return attended_features

def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
    since = time.time()
    
    Loss_list = {'train': [], 'valid': []}
    Accuracy_list = {'train': [], 'valid': []}
    
    best_model_wts = model.state_dict()
    best_loss = float('inf')
    counter = 0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = []
            running_corrects = []

            for inputs, labels, image_id in dataloaders[phase]:
                if torch.cuda.is_available():
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                score = accuracy_metric(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss.append(loss.item())
                running_corrects.append(score.item())

            epoch_loss = np.mean(running_loss)
            epoch_acc = np.mean(running_corrects)

            print('{} Loss: {:.4f} IoU: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            Loss_list[phase].append(epoch_loss)
            Accuracy_list[phase].append(epoch_acc)

            if phase == 'valid' and epoch_loss <= best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()
                counter = 0
                if epoch > 50:
                    torch.save(model.state_dict(), f'save_models/epoch_{epoch}_{epoch_acc}.pth')
            elif phase == 'valid' and epoch_loss > best_loss:
                counter += 1
            if phase == 'train':
                scheduler.step()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    model.load_state_dict(best_model_wts)
    return model, Loss_list, Accuracy_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='data/', help='the path of images')
    parser.add_argument('--csvfile', type=str, default='src/test_train_data.csv', help='two columns [image_id,category(train/test)]')
    parser.add_argument('--loss', default='dice', help='loss type')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epoch', type=int, default=150, help='epoches')
    args = parser.parse_args()

    os.makedirs(f'save_models/', exist_ok=True)

    df = pd.read_csv(args.csvfile)
    df = df[df.category=='train']
    df.reset_index(drop=True, inplace=True)
    gkf = GroupKFold(n_splits=5)
    df['fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=df.image_id.tolist())):
        df.loc[val_idx, 'fold'] = fold

    fold = 0
    val_files = list(df[df.fold==fold].image_id)
    train_files = list(df[df.fold!=fold].image_id)

    train_dataset = binary_class(args.dataset, train_files, get_train_transform())
    val_dataset = binary_class(args.dataset, val_files, get_valid_transform())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch//2, drop_last=True)

    dataloaders = {'train': train_loader, 'valid': val_loader}

    model_ft = DCSAU_Net.Model(img_channels = 3, n_classes = 1)



    if torch.cuda.is_available():
        model_ft = model_ft.cuda()

    # Loss, IoU and Optimizer
    if args.loss == 'ce':
        criterion = nn.BCELoss()
    if args.loss == 'dice':
        criterion = DiceLoss_binary()

    accuracy_metric = IoU_binary()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=args.lr)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.5)

    model_ft, Loss_list, Accuracy_list = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=args.epoch)

    torch.save(model_ft, f'save_models/epoch_last.pth')

    plt.title('Validation loss and IoU')
    valid_data = pd.DataFrame({'Loss': Loss_list["valid"], 'IoU': Accuracy_list["valid"]})
    valid_data.to_csv(f'valid_data.csv')
    sns.lineplot(data=valid_data, dashes=False)
    plt.ylabel('Value')
    plt.xlabel('Epochs')
    plt.savefig('valid.png')

    plt.figure()
    plt.title('Training loss and IoU')
    train_data = pd.DataFrame({'Loss': Loss_list["train"], 'IoU': Accuracy_list["train"]})
    train_data.to_csv(f'train_data.csv')
    sns.lineplot(data=train_data, dashes=False)
    plt.ylabel('Value')
    plt.xlabel('Epochs')
    plt.savefig('train.png')
