import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

def functional_conv(x, weights, biases, bn_weights, bn_biases):
    x = F.conv2d(x, weights, biases, padding=1)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=bn_weights, bias=bn_biases, training=True)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2)
    return x

class MAML(nn.Module):
    def __init__(self):
        super(MAML, self).__init__()
        self.conv1 = conv_block(3, 32)
        self.conv2 = conv_block(32, 32)
        self.conv3 = conv_block(32, 32)
        self.conv4 = conv_block(32, 32)
        self.logits = nn.Linear(800, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.x.view(x.size(0), -1)
        return self.logits(x)

    def adaptation(self, x, weights):
        for block in [1, 2, 3, 4]:
            x = functional_conv(x, weights[f'conv{block}.0.weight'], weights[f'conv{block}.0.bias'],
                                weights.get(f'conv{block}.1.weight'), weights.get(f'conv{block}.1.bias'))
        
        x = x.view(x.size(0), -1)
        return F.linear(x, weights['logits.weight'], weights['logits.bias'])