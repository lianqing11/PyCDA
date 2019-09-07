import torch
import torch.nn as nn
from lib.nn import SynchronizedBatchNorm2d
import torch.nn.functional as F


def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        # print m.__class__.__name__
        m.weight.data.normal_(0.0, 0.02)

class INSResBlock(nn.Module):
  def conv3x3(self, inplances, out_plances, stride=1):
      return nn.Conv2d(inplances, out_plances, kernel_size=3, stride=stride, padding=1)

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





class UNIT(nn.Module):
    def __init__(self, input_dim = 2048):
        super(UNIT, self).__init__()
        ch = 256
        enc = []
        tch = ch
        enc += [LeakyReLUConv2d(input_dim, ch, kernel_size=7, stride=1, padding=3)]
        dec = []
        for i in range(1):
            dec += [INSResBlock(tch, tch)]
        for i in range(3):
            dec += [LeakyReLUConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
            tch = tch // 2

        dec += [LeakyReLUConv2d(tch, 3, kernel_size=1, stride=1, padding=0)]
        dec += [nn.Tanh()]

        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



