from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
#from torchvision import functional as F
import torch.functional as F

from torch.autograd import Variable
import numpy as np

import time


# some constants defined here for the best model
#-----------------------------------------------
# for general model
noise_size = 100
ngf = 128
ndf = 128
image_channels = 3
ker = 4
pad = int((ker - 2)/2)
strd = 2
relu_leak = 0.2
prob = 0.0 # no dropout

# for residual unit
res_ker = 3
res_pad = 1
res_strd = 1
#-----------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, channel_size):
        super(ResidualBlock, self).__init__()
        
        self.main = nn.Sequential(
            nn.Conv2d(channel_size, channel_size, res_ker, res_strd, res_pad, bias=False),
            nn.BatchNorm2d(channel_size),
            nn.ReLU(True),
            nn.Conv2d(channel_size, channel_size, res_ker, res_strd, res_pad, bias=False),
            nn.BatchNorm2d(channel_size)
        ) 
        self.res_relu = nn.ReLU(True)

    def forward(self, inp):
        res = inp # first store the residual value
        output = self.main(inp)
        output += res # residual connection
        output = self.res_relu(output)
        return output


class Generator_Stage1(nn.Module):
    def __init__(self):
        super(Generator_Stage1, self).__init__()

        self.main = nn.Sequential(
            # 1 -> 4
            nn.ConvTranspose2d(noise_size, ngf * 8, ker, 1, pad-1, bias=False),
            nn.ReLU(True),
            # 4 -> 8
            nn.ConvTranspose2d(ngf*8, ngf * 4, ker, strd, pad, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.Dropout2d(prob),
            nn.ReLU(True),
            # 8 -> 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, ker, strd, pad, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.Dropout2d(prob),
            nn.ReLU(True),
            # 16 -> 32
            nn.ConvTranspose2d(ngf * 2, ngf, ker, strd, pad, bias=False),
            nn.BatchNorm2d(ngf),
            nn.Dropout2d(prob),
            nn.ReLU(True),
            # 32 -> 64
            nn.ConvTranspose2d(ngf, image_channels, ker, strd, pad, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output

class Discriminator_Stage1(nn.Module):
    def __init__(self):
        super(Discriminator_Stage1, self).__init__()

        self.main = nn.Sequential(
            # 64 -> 32
            nn.Conv2d(image_channels, ndf, ker, strd, pad, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 -> 16
            nn.Conv2d(ndf, ndf * 2, ker, strd, pad, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(relu_leak, inplace=True),
            # 16 -> 8
            nn.Conv2d(ndf * 2, ndf * 4, ker, strd, pad, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(relu_leak, inplace=True),
            # 8 -> 4
            nn.Conv2d(ndf * 4, ndf * 8, ker, strd, pad, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(relu_leak, inplace=True),
            # 4 -> 1
            nn.Conv2d(ndf * 8, 1, ker, 1, pad-1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, in_x):
        output = self.main(in_x)
        return output.view(-1, 1).squeeze(1) # (N,1) output flattened to (N,)

class Generator_Stage2(nn.Module):
    def __init__(self):
        super(Generator_Stage2, self).__init__()

        # 64 -> 32
        self.preprocessing = nn.Sequential(
            nn.Conv2d(image_channels, ngf, ker, strd, pad, bias=False),
            nn.ReLU(True)
        )
        # residuals
        self.residual = nn.Sequential(
            ResidualBlock(ngf),
            ResidualBlock(ngf),
            ResidualBlock(ngf),
            ResidualBlock(ngf),
            ResidualBlock(ngf),
            ResidualBlock(ngf),
        )
        self.ending_residual = nn.Sequential(
            nn.Conv2d(ngf, ngf, res_ker, res_strd, res_pad, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # at this part, add the residual inputs from after the preprocessing

        scale_factor = 2 # upscaling should be factor of 2 increase
        mode = 'nearest' # upscaling method is nearest-neighbour
        self.main = nn.Sequential(
            # 32 -> 64
            nn.Upsample(scale_factor=scale_factor, mode=mode),
            nn.Conv2d(ngf, ngf*4, res_ker, res_strd, res_pad, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            # 64 -> 128
            nn.Upsample(scale_factor=scale_factor, mode=mode),
            nn.Conv2d(ngf*4, image_channels, res_ker, res_strd, res_pad, bias=False),
            nn.Tanh()
        )

    def forward(self, in_x):
        x_p = self.preprocessing(in_x)
        x_r = x_p
        x_r = self.residual(x_r)
        x_r = self.ending_residual(x_r)
        # large residual connections
        x_f = x_r + x_p
        x_f = self.main(x_f)
        return x_f

class Discriminator_Stage2(nn.Module):
    def __init__(self):
        super(Discriminator_Stage2, self).__init__()

        self.main = nn.Sequential(
            # 128->64
            nn.Conv2d(image_channels, ndf, ker, strd, pad, bias=False),
            nn.LeakyReLU(relu_leak, inplace=True),
            # 64->32
            nn.Conv2d(ndf, ndf * 2, ker, strd, pad, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(relu_leak, inplace=True),
            # 32->16
            nn.Conv2d(ndf * 2, ndf * 2, ker, strd, pad, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(relu_leak, inplace=True),
            # 16->8
            nn.Conv2d(ndf * 2, ndf * 4, ker, strd, pad, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(relu_leak, inplace=True),
            # 8->4
            nn.Conv2d(ndf * 4, ndf * 8, ker, strd, pad, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(relu_leak, inplace=True),

            # 4->1
            nn.Conv2d(ndf * 8, 1, ker, 1, pad-1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, in_x):
        output = self.main(in_x).view(-1, 1).squeeze(1) # (N,1) output flattened to (N,)
        return output
