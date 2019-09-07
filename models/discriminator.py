import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from lib.nn import SynchronizedBatchNorm2d
import math

class GradReverse(torch.autograd.Function):
    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return grad_output.neg()



class _ConvBatchNormReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, relu=True, need_bn=False):
        super(_ConvBatchNormReLU, self).__init__()
        self.add_module("conv",
             nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
        )
        if need_bn:
            self.add_module("bn", SynchronizedBatchNorm2d(out_channels))
        if relu:
            self.add_module("relu", nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, x):
        return super(_ConvBatchNormReLU, self).forward(x)



class AsppModule(nn.Module):
    """Atrous Spatial Pyramid Pooling with image pool"""

    def __init__(self, in_channels, out_channels, pyramids, with_relu=True, need_bn=False):
        super(AsppModule, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module(
            "c0", _ConvBatchNormReLU(in_channels, out_channels, 1, 1, 0, 1, relu=with_relu, need_bn=need_bn)
        )
        for i, (dilation, padding) in enumerate(zip(pyramids, pyramids)):
            self.stages.add_module(
                "c{}".format(i + 1),
                _ConvBatchNormReLU(in_channels, out_channels, 3, 1, padding, dilation, relu=with_relu),
            )

    def forward(self, x):
        h = []
        for stage in self.stages.children():
            h += [stage(x)]
        h = torch.cat(h, dim=1)
        return h

class PureDialtedDiscriminator(nn.Module):
    def __init__(self,  input_dim, reverse_grad=False):
        super(PureDialtedDiscriminator, self).__init__()
        self.reverse_grad = reverse_grad
        if self.reverse_grad == True:
            self.reverse_layer = GradReverse()
        self.aspp = AsppModule(input_dim, 128, [2,3,4], with_relu=False)
        self.conv2 = nn.Conv2d(128*4, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = x[-1]
        if self.reverse_grad == True:
            x = self.reverse_layer(x)
        x = self.aspp(x)
        x = self.conv2(x)
        return x, None


class DialtedDiscriminator(nn.Module):
    def __init__(self,  input_dim, reverse_grad=False):
        super(DialtedDiscriminator, self).__init__()
        self.reverse_grad = reverse_grad
        if self.reverse_grad == True:
            self.reverse_layer = GradReverse()
        self.conv1 = nn.Conv2d(input_dim, 128, kernel_size=1, stride=1, padding=0)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.aspp = AsppModule(128, 128, [2,3,4])
        self.conv2 = nn.Conv2d(128*4, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = x[-1]
        if self.reverse_grad == True:
            x = self.reverse_layer(x)
        x = self.conv1(x)
        x = self.leaky_relu1(x)
        x = self.aspp(x)
        x = self.conv2(x)
        return x, None

class DialtedDiscriminator_no_relu(nn.Module):
    def __init__(self,  input_dim, reverse_grad=False):
        super(DialtedDiscriminator_no_relu, self).__init__()
        self.reverse_grad = reverse_grad
        if self.reverse_grad == True:
            self.reverse_layer = GradReverse()
        self.conv1 = nn.Conv2d(input_dim, 128, kernel_size=1, stride=1, padding=0)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.aspp = AsppModule(128, 128, [2,3,4], with_relu=False)
        self.conv2 = nn.Conv2d(128*4, 1, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x = x[-1]
        if self.reverse_grad == True:
            x = self.reverse_layer(x)
        x = self.conv1(x)
        x = self.leaky_relu1(x)
        x = self.aspp(x)
        x = self.conv2(x)
        return x, None



class CycleD(nn.Module):
    def __init__(self, input_dim=4096, reverse_grad=False):
        super(CycleD, self).__init__()
        self.reverse_grad = reverse_grad
        if self.reverse_grad == True:
            self.reverse_layer = GradReverse()
        dim1 = 1024 if input_dim==4096 else 512
        dim2 = int(dim1/2)
        self.D = nn.Sequential(
            nn.Conv2d(input_dim, dim1, 1),
            nn.Dropout2d(p=0.5),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim1, dim2, 1),
            nn.Dropout2d(p=0.5),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim2, 1, 1)
        )
    def forward(self, x): 
        x = x[-1]
        if self.reverse_grad == True:
            x = self.reverse_layer(x)
        d_score = self.D(x)
        return d_score, None



class AdaptSeg_D(nn.Module):
    def __init__(self, input_dim, reverse_grad=False, ndf = 64):
        super(AdaptSeg_D, self).__init__()
        self.reverse_grad = reverse_grad
        if self.reverse_grad == True:
            self.reverse_layer = GradReverse()
        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)


    def forward(self, x):
        x = x[-1]
        if self.reverse_grad==True:
            x = self.reverse_layer(x)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)

        return x, None



class DialtedDiscriminator_sub_bn(nn.Module):
    def __init__(self,  input_dim, reverse_grad=False):
        super(DialtedDiscriminator_sub_bn, self).__init__()
        self.reverse_grad = reverse_grad
        if self.reverse_grad == True:
            self.reverse_layer = GradReverse()
        self.conv1 = nn.Conv2d(input_dim, 128, kernel_size=1, stride=1, padding=0)
        self.bn1 = SynchronizedBatchNorm2d(128)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.aspp = AsppModule(128, 128, [2,3,4])
        self.conv2 = nn.Conv2d(128*4, 1, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        x = x[-1]
        if self.reverse_grad == True:
            x = self.reverse_layer(x)
        x = self.conv1(x)
        x = self.leaky_relu1(x)
        x = self.aspp(x)
        x = self.conv2(x)
        return x, None


class DialtedDiscriminator_bn(nn.Module):
    def __init__(self,  input_dim, reverse_grad=False):
        super(DialtedDiscriminator_bn, self).__init__()
        self.reverse_grad = reverse_grad
        if self.reverse_grad == True:
            self.reverse_layer = GradReverse()
        self.aspp = AsppModule(input_dim, 128, [2,3,4], with_relu=True, need_bn=True)
        self.conv2 = nn.Conv2d(128*4, 1, kernel_size=1, stride=1, padding=0)



    def forward(self, x):
        x = x[-1]
        if self.reverse_grad == True:
            x = self.reverse_layer(x)
        x = self.aspp(x)
        x = self.conv2(x)
        return x, None
