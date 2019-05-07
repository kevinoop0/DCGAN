# -*- coding:utf-8 -*-
import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self,channel):
        super(ResidualBlock,self).__init__()
        self.conv1 = nn.Conv2d(channel,channel,kernel_size=5,stride=1,padding=2)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu =  nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channel,channel,kernel_size=5,stride=1,padding=2)
        self.bn2 = nn.BatchNorm2d(channel)

    def forward(self, x):
        res = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + res
        return out

class ResidualBlock_deconv(nn.Module):
    def __init__(self,channel):
        super(ResidualBlock_deconv,self).__init__()
        self.conv1 = nn.ConvTranspose2d(channel,channel,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu = nn.ELU()
        self.conv2 = nn.ConvTranspose2d(channel,channel,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(channel)

    def forward(self, x):
        res = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + res
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels,kernel_size,stride,padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride,padding=padding)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels,  kernel_size=kernel_size, stride=stride,padding=padding)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return out


class NetG(nn.Module):
    """
    生成器定义
    """
    def __init__(self, opt):
        super(NetG, self).__init__()
        ngf = opt.ngf  # 生成器feature map数

        self.main = nn.Sequential(
            # 输入是一个nz维度的噪声，我们可以认为它是一个1*1*nz的feature map
            #100 *1*1
            nn.ConvTranspose2d(opt.nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # 上一步的输出形状：(ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            ResidualBlock_deconv(ngf * 4),
            # 上一步的输出形状： (ngf*4) x 8 x 8

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            ResidualBlock_deconv(ngf * 2),
            # 上一步的输出形状： (ngf*2) x 16 x 16

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            ResidualBlock_deconv(ngf),
            # 上一步的输出形状：(ngf) x 32 x 32

            nn.ConvTranspose2d(ngf, 3, 5, 3, 1, bias=False),
            nn.Tanh()  # 输出范围 -1~1 故而采用Tanh
            # 输出形状：3 x 96 x 96
        )

    def forward(self, input):
        return self.main(input)


class NetD(nn.Module):
    """
    判别器定义
    """
    def __init__(self, opt):
        super(NetD, self).__init__()
        ndf = opt.ndf
        self.main = nn.Sequential(
            # 输入 3 x 96 x 96
            nn.Conv2d(3, ndf, 5, 3, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # ResidualBlock(ndf, 5, 1, 2),
            # 输出 (ndf) x 32 x 32

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(ndf * 2, 5, 1, 2),
            # 输出 (ndf*2) x 16 x 16

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(ndf * 4, 5, 1, 2),
            # 输出 (ndf*4) x 8 x 8

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(ndf * 8, 5, 1, 2),
            # 输出 (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # 输出一个数(概率)
        )

    def forward(self, input):
        return self.main(input).view(-1)
