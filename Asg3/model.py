#
# CSCI3290 Computational Imaging and Vision *
# --- Declaration --- *
# I declare that the assignment here submitted is original except for source
# material explicitly acknowledged. I also acknowledge that I am aware of
# University policy and regulations on honesty in academic work, and of the
# disciplinary guidelines and procedures applicable to breaches of such policy
# and regulations, as contained in the website
# http://www.cuhk.edu.hk/policy/academichonesty/ *
# Assignment 3
# Name : Li Chi
# Student ID : 1155172017
# Email Addr : 1155172017@link.cuhk.edu.hk
#

import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        # First convolutional layer (Patch Extraction)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Second convolutional layer (Non-linear Mapping)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Third convolutional layer (Reconstruction)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, padding=2, stride=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=3, mode='bicubic', align_corners=True )
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)  # No activation function (identity)
        return x
