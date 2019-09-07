import torch
import torch.nn as nn
from lib.nn import SynchronizedBatchNorm2d


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=49, stride=stride,
                     padding=24, bias=False)


class BasicRefine(nn.Module):
    def __init__(self, inplanes, planes):
        super(BasicRefine, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=1)

    def forward(self, x):
        return self.conv1(x)



class Refine2(nn.Module):
    def __init__(self, inplanes, planes):
        super(Refine2, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=1)
        self.bn1 = SynchronizedBatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x



