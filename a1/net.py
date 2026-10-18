"""
   net.py
   COMP9444, CSE, UNSW
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class NetLin(nn.Module):
    # linear network with no activation function
    def __init__(self):
        super(NetLin, self).__init__()
        self.in_out = nn.Linear(784, 10)

    def forward(self, x):
        flat = x.reshape(x.shape[0], 784)
        return self.in_out(flat)

    def viz(self): # create and save images
        weights = self.in_out.weight.data
        for k in range(10):
            img = min_max_scale(weights[k].reshape(28, 28))
            torchvision.transforms.functional.to_pil_image(img).save('in_out'+str(k)+'.png')

class NetSkip(nn.Module):
    # network combining direct input-to-output weights
    # with a fully connected hidden layer of 12 nodes
    def __init__(self):
        super(NetSkip, self).__init__()
        self.in_out = nn.Linear(784, 10)
        self.in_hid = nn.Linear(784, 12)
        self.hid_out = nn.Linear(12, 10)

    def forward(self, x):
        flat = x.reshape(x.shape[0], 784)
        hid  = F.relu(self.in_hid(flat))
        return self.in_out(flat) + self.hid_out(hid)

    def viz(self): # create and save images
        w1 = self.in_hid.weight.data
        w2 = self.hid_out.weight.data

        for k in range(12):
            img = min_max_scale(w1[k].reshape(28, 28))
            torchvision.transforms.functional.to_pil_image(img).save('in_hid'+str(k)+'.png')

        img2 = min_max_scale(w2)
        torchvision.transforms.functional.to_pil_image(img2).save('hid_out.png')
        

class NetConv(nn.Module):
    # two fully convolutional layers using relu,
    # one fully connected (hidden) layer using relu,
    # and one fully connected (output) layer
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=8, stride=2)    # 28x28 -> 6 x 11x11
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=2)   # 11x11 -> 16 x 4x4
        self.fc1 = nn.Linear(256, 96)                            # 16*4*4 = 256
        self.fc2 = nn.Linear(96, 10)
 
    def forward(self, x):
        x = x.view(x.shape[0], 1, 28, 28)       # ensure [b, 1, 28, 28]
        x = F.relu(self.conv1(x))               # [b, 6, 11, 11]
        x = F.relu(self.conv2(x))               # [b, 16, 4, 4]
        flat = x.view(x.shape[0], -1)           # [b, 256]
        hid = F.relu(self.fc1(flat))            # [b, 96]
        return self.fc2(hid)                    # [b, 10]
 
    def viz(self): # create and save images
        weights = self.conv1.weight.data        # [6, 1, 8, 8]
        for h in range(6):
            img = min_max_scale(weights[h].reshape(8, 8))
            img = torchvision.transforms.functional.to_pil_image(img)
            img = img.resize((8 * 30, 8 * 30), resample=0)  # enlarge for readability
            img.save('filt' + str(h) + '.png')


def min_max_scale(w):
    # scale tensor values linearly to the range [0,1]
    w_min = w.min()
    w_max = w.max()
    eps = 0.00000001
    return((w - w_min)/(w_max - w_min + eps))
