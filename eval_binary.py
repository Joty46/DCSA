import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from data_loading import binary_class
import albumentations as A
from albumentations.pytorch import ToTensor
import torch.nn.functional as F
import argparse
import time
import pandas as pd
import cv2
import os
from skimage import io, transform
from PIL import Image


class IoU(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        IoU = (intersection + smooth)/(union + smooth)
        return IoU

class Dice(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Dice, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        return dice

class F1Score(nn.Module):
    def __init__(self, smooth=1):
        super(F1Score, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        true_positive = (predictions * targets).sum()
        false_positive = ((1 - targets) * predictions).sum()
        false_negative = (targets * (1 - predictions)).sum()

        precision = true_positive / (true_positive + false_positive + self.smooth)
        recall = true_positive / (true_positive + false_negative + self.smooth)

        f1 = (2 * precision * recall) / (precision + recall + self.smooth)

        return f1



def get_transform():
    return A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()
        ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data/', type=str, help='the path of dataset')
    parser.add_argument('--csvfile', default='src/test_train_data.csv', type=str, help='two columns [image_id,category(train/test)]')
    parser.add_argument('--model', default='save_models/epoch_last.pth', type=str, help='the path of model')
    parser.add_argument('--debug', default=True, type=bool, help='plot mask')
    args = parser.parse_args()

    os.makedirs('debug/', exist_ok=True)

    df = pd.read_csv(args.csvfile)
    df = df[df.category == 'test']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_files = list(df.image_id)
    test_dataset = binary_class(args.dataset, test_files, get_transform())
    

    model = torch.load(args.model)
    model = model.cuda()

    iou_eval = IoU()
    dice_eval = Dice()
    f1_metric = F1Score()
    time_cost = []

    since = time.time()

    for image_id in test_files:
        if(image_id != '.ipynb_checkpoints'):
          img = cv2.imread(f'/content/drive/MyDrive/DCSAU-Net_Lung/data/images/{image_id}')
          #print(image_id);
          img = cv2.resize(img, ((256, 256)))
          img_id = list(image_id.split('.'))[0]
          cv2.imwrite(f'debug/{img_id}.png', img)

    with torch.no_grad():
        for img, mask, img_id in test_dataset:
          if(img_id != '.ipynb_checkpoints'):
            print(img.shape)
            img = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False).cuda()
            mask = Variable(torch.unsqueeze(mask, dim=0).float(), requires_grad=False).cuda()
            torch.cuda.synchronize()
            start = time.time()
            pred = model(img)
            torch.cuda.synchronize()
            end = time.time()
            time_cost.append(end - start)

            pred = torch.sigmoid(pred)

            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0

            pred_draw = pred.clone().detach()
            mask_draw = mask.clone().detach()

            if args.debug:
                if(image_id != '.ipynb_checkpoints'):
                  img_id = list(img_id.split('.'))[0]
                  img_numpy = pred_draw.cpu().detach().numpy()[0][0]
                  img_numpy[img_numpy == 1] = 255
                  cv2.imwrite(f'debug/{img_id}_pred.png', img_numpy)

                  mask_numpy = mask_draw.cpu().detach().numpy()[0][0]
                  mask_numpy[mask_numpy == 1] = 255
                  cv2.imwrite(f'debug/{img_id}_gt.png', mask_numpy)

            iouscore = iou_eval(pred, mask)
            dicescore = dice_eval(pred, mask)
            f1score = f1_metric(pred, mask)

            print('IoU:', iouscore.item())
            print('Dice:', dicescore.item())
            print('F1 Score:', f1score.item())

            torch.cuda.empty_cache()

    time_elapsed = time.time() - since

    print('Evaluation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('FPS: {:.2f}'.format(1.0 / (sum(time_cost) / len(time_cost))))
