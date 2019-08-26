import os
from math import ceil
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from PIL import Image, ImageDraw, ImageFont
from skimage.feature import canny
from torch.utils import data
import nibabel as nib
#read image, normalized, convert to pytorch tensor
class Dataset(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, data_path, pairs):
        "Initialization"
        self.data_path = data_path
        self.pairs = pairs
    def __len__(self):
        "Denotes the total number of samples"
        return len(self.pairs)
    def normalize(self, X):
        X_norm = (X - np.min(X)) / (np.max(X) - np.min(X))
        return X_norm
    def __getitem__(self, index):
        "Generates one sample of data"
        i, j = self.pairs[index]
        img_file = os.path.join(self.data_path, 'volume-{}.nii'.format(i))
        seg_file = os.path.join(self.data_path, 'segmentation-{}.nii'.format(i))
        # Load input data: (vol i, slice j)
        X = nib.load(img_file).get_fdata()[:, :, j]  # input images
        X = self.normalize(X)
        y = nib.load(seg_file).get_fdata()[:, :, j]  # labels
        # convert to Torch tensor
        X = torch.FloatTensor(np.expand_dims(X, axis=0)).repeat(3, 1, 1)  # dim=(ch, x_size, y_size)
        y = torch.LongTensor(y)
        return X, y
# check if directory exists, if not make it
def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
#initialization of weights in NN
def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            #for convolutional layer and linear layer
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            # for batchnomarlization
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
"""  
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[list(range(in_channels)), list(range(out_channels)), :, :] = filt
    return torch.from_numpy(weight).float()
"""
##label true for each pixel, 0, 1, 2
##label_pred for each pixel is one-hot, (0,0,1) (0,1,0) (0,0,1)
def _fast_confusion(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    confusion = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return confusion
def DSC(y_pred_batch, y_batch, num_classes):
    # confusion matrix
    DSCs = []
    # confusion = np.zeros((num_classes, num_classes))     # compute each sample
    for lp, lt in zip(y_pred_batch, y_batch):
        confusion = _fast_confusion(lp.flatten(), lt.flatten(), num_classes)
        # compute each sample DSC = 2 (|X n Y|)/ (|X| + |Y|)
        # dsc has num_classes of elements, which is dice for each class
        dsc = 2 * np.diag(confusion) / (confusion.sum(axis=1) + confusion.sum(axis=0))
        DSCs.append(dsc)
    DSCs = np.array(DSCs)    # size = (N_batch, num_class)
    return DSCs
def evaluate(y_pred_all, y_all, num_classes):
    # confusion matrix
    DSCs = []
    confusion = np.zeros((num_classes, num_classes))     # compute each sample
    batch_confusion = np.zeros((num_classes, num_classes))  # compute one batch
    for lp, lt in zip(y_pred_all, y_all):
        confusion = _fast_confusion(lp.flatten(), lt.flatten(), num_classes)
        batch_confusion += _fast_confusion(lp.flatten(), lt.flatten(), num_classes)
    # confusion matrix, axis 0: y (labels), axis 1: y_pred
    # pixel accuracy
    acc = np.diag(batch_confusion).sum() / batch_confusion.sum()
    # what portion of pixels of class X is successfully classified as class X
    acc_cls = np.diag(batch_confusion) / batch_confusion.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    #intersection over union |X n Y|/|X u Y|
    iu = np.diag(batch_confusion) / (batch_confusion.sum(axis=1) + batch_confusion.sum(axis=0) - np.diag(batch_confusion))
    mean_iu = np.nanmean(iu)    # avg over all classes
    # freq is the ratio of pixels in each class
    freq = batch_confusion.sum(axis=1) / batch_confusion.sum()
    # average iu over class weighted by its freq
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc, DSCs
#calculate average loss over batches
class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.new_avg_loss = 0
        self.avg = 0
        self.sum = 0
        self.N= 0
    def update(self, new_avg_loss, N_batch):
        self.new_loss = new_avg_loss
        self.sum += new_avg_loss * N_batch
        self.N += N_batch
        self.avg = self.sum / self.N
#define the block in contraction path
class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)
    def forward(self, x):
        return self.encode(x)
#define the block in expansion path
class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )
    def forward(self, x):
        return self.decode(x)
# U-Net
class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        #contraction path
        self.enc1 = _EncoderBlock(3, 64)
        self.enc2 = _EncoderBlock(64, 128)
        self.enc3 = _EncoderBlock(128, 256)
        self.enc4 = _EncoderBlock(256, 512, dropout=True)
        #bottle neck #input is the concatenation 256+256=512
        self.center = _DecoderBlock(512, 1024, 512)
        #expansion pathof output of last layer and of contraction path
        self.dec4 = _DecoderBlock(1024, 512, 256)
        self.dec3 = _DecoderBlock(512, 256, 128)
        self.dec2 = _DecoderBlock(256, 128, 64)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #need conv 1X1 to output contouring map
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        initialize_weights(self)
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(enc4)
        #interpolation is to make the two concatenated inputs of the same dimension
        # start from 2, since any intermediate output of size (batch_number, chanel number, ...)
        dec4 = self.dec4(torch.cat([center, F.interpolate(enc4, center.size()[2:], mode='bilinear', align_corners=True)], 1))
        dec3 = self.dec3(torch.cat([dec4, F.interpolate(enc3, dec4.size()[2:], mode='bilinear', align_corners=True)], 1))
        dec2 = self.dec2(torch.cat([dec3, F.interpolate(enc2, dec3.size()[2:], mode='bilinear', align_corners=True)], 1))
        dec1 = self.dec1(torch.cat([dec2, F.interpolate(enc1, dec2.size()[2:], mode='bilinear', align_corners=True)], 1))
        final = self.final(dec1)
        return F.interpolate(final, x.size()[2:], mode='bilinear', align_corners=True)
        print("\tIn Model: input size", x.size(),
              "output size", x.size())