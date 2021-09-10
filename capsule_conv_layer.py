import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
# from torch import regularizers

class CapsuleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size,stride):
        super(CapsuleConvLayer, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, # fixme constant
                               stride=stride,
                               padding=0,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv0(x))