
"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import cv2
import numpy as np
import torch.nn.functional as F

def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        # print m.__class__.__name__
        m.weight.data.normal_(0.0, 0.02)

def xavier_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0.1)



class GaussianSmoother(nn.Module):
  def __init__(self, kernel_size=5):
    super(GaussianSmoother, self).__init__()
    self.sigma = 0.3*((kernel_size-1)*0.5-1)+0.8
    kernel = cv2.getGaussianKernel(kernel_size, -1)
    kernel2d = np.dot(kernel.reshape(kernel_size,1),kernel.reshape(1,kernel_size))
    data = torch.Tensor(3, 1, kernel_size, kernel_size)
    self.pad = (kernel_size-1)/2
    for i in range(0,3):
      data[i,0,:,:] = torch.from_numpy(kernel2d)
    self.blur_kernel = Variable(data, requires_grad=False)

  def forward(self, x):
    out = nn.functional.pad(x, [self.pad, self.pad, self.pad, self.pad], mode ='replicate')
    out = nn.functional.conv2d(out, self.blur_kernel, groups=3)
    return out

  def cuda(self, gpu):
    self.blur_kernel = self.blur_kernel.cuda(gpu)

class INSResBlock(nn.Module):
  def conv3x3(self, inplanes, out_planes, stride=1):
    return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1)

  def __init__(self, inplanes, planes, stride=1, dropout=0.0):
    super(INSResBlock, self).__init__()
    model = []
    model += [self.conv3x3(inplanes, planes, stride)]
    model += [nn.InstanceNorm2d(planes)]
    model += [nn.LeakyReLU(inplace=True)]
    model += [self.conv3x3(planes, planes)]
    model += [nn.InstanceNorm2d(planes)]
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)

  def forward(self, x):
    residual = x
    out = self.model(x)
    #print(x.size(), residual.size())
    out += residual
    return out


class LeakyReLUConv2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
    super(LeakyReLUConv2d, self).__init__()
    model = []
    model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)]
    model += [nn.LeakyReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)

  def forward(self, x):
    return self.model(x)

class LeakyReLUConvTranspose2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding=0, output_padding=0):
    super(LeakyReLUConvTranspose2d, self).__init__()
    model = []
    model += [nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=True)]
    model += [nn.LeakyReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)

  def forward(self, x):
      #print(type(x))
      return self.model(x)

class AutoEncoder(nn.Module):
    def __init__(self, input_dim_a = 2048):
        super(AutoEncoder, self).__init__()
        ch = 256
        enc = []
        tch = ch
        enc += [LeakyReLUConv2d(input_dim_a, ch, kernel_size=7, stride=1, padding=3)]
        #for i in range(1, 4):
        #    enc += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
        #    tch *=2
        #enc += [LeakyReLUConv2d(input_dim_a, tch, kernel_size=1, stride=1, padding=1)]
        #enc_shared = []
        #for i in range(0, 1):
        #    enc_shared += [INSResBlock(tch, tch)]
        #enc_shared += [GaussianNoiseLayer()]
        dec_shared = []
        for i in range(0, 1):
            dec_shared += [INSResBlock(tch, tch)]
        dec_A = []
        dec_B = []
        for i in range(0, 3):
            dec_A += [INSResBlock(tch, tch)]
            dec_B += [INSResBlock(tch, tch)]
        for i in range(0, 4-1):
            dec_A += [LeakyReLUConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
            dec_B += [LeakyReLUConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
            tch = tch // 2
        #dec += [LeakyReLUConvTranspose2d(tch, tch, kernel_size=3, stride=2, padding=1, output_padding=1)]
        dec_A += [LeakyReLUConv2d(tch, 3, kernel_size=1, stride=1, padding=0)]
        dec_A += [nn.Tanh()]
        dec_B += [LeakyReLUConv2d(tch, 3, kernel_size=1, stride=1, padding=0)]
        dec_B += [nn.Tanh()]
        self.encode = nn.Sequential(*enc)
        #self.enc_shared = nn.Sequential(*enc_shared)
        self.dec_shared = nn.Sequential(*dec_shared)
        self.decode_A = nn.Sequential(*dec_A)
        self.decode_B = nn.Sequential(*dec_B)


    def forward(self, x, which):
        out = self.encode(x)
        #out = self.enc_shared(out)
        #result = self.reparameterize(mu, log_var)
        out = self.dec_shared(out)
        if which == 'source':
            result = self.decode_A(out)
        else:
            result = self.decode_B(out)
        return result

