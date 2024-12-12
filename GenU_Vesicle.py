import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import os
from tqdm import tqdm
class UNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(input_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.bottleneck = self.conv_block(256, 512)
        self.up3 = self.upconv_block(512, 256)
        self.dec3 = self.conv_block(512, 256)  # Skip connection
        self.up2 = self.upconv_block(256, 128)
        self.dec2 = self.conv_block(256, 128)
        self.up1 = self.upconv_block(128, 64)
        self.dec1 = self.conv_block(128, 64)
        self.final = nn.Conv2d(64, output_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        bottleneck = self.bottleneck(F.max_pool2d(enc3, 2))
        dec3 = self.up3(bottleneck)
        dec3 = self.dec3(torch.cat((dec3, enc3), dim=1))
        dec2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat((dec2, enc2), dim=1))
        dec1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat((dec1, enc1), dim=1))
        return torch.sigmoid(self.final(dec1))

model = UNet()
model.load_state_dict(torch.load('./GenU_Vesicle_seg.pth',map_location=torch.device('cpu')))
model.eval()
img = cv2.imread(r"C:\Users\guozh\Desktop\segmentation\v.tif",cv2.IMREAD_UNCHANGED).astype('float32')
ori_shape = img.shape[0]
img = cv2.resize(img,(2048,2048),cv2.INTER_LINEAR)
img = img[np.newaxis, :, :]
img = torch.tensor(img)
img = img.to('cpu').unsqueeze(0)
with torch.no_grad():
    mask = model(img).detach( ).cpu().squeeze(0).reshape(2048,2048)
mask = cv2.resize(np.array(mask),(ori_shape,ori_shape),cv2.INTER_LINEAR)
plt.imshow(mask>0.2,cmap='gray')
plt.show(block=True)
mask = mask>0.2
mask = mask.astype('int')
cv2.imwrite('./vesicle_mask.tif',mask)
