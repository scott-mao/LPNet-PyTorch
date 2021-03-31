import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from model import common
import numpy as np
def make_model(args, parent=False):
    return LPNet(args)

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    x = F_variance.pow(0.5)
    return x

# contrast-aware channel attention module
class CCALayer(nn.Module):
    def __init__(self, args, reduction=16):
        super(CCALayer, self).__init__()
        channel = args.n_feats
        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class MSCFB(nn.Module):
    def __init__(self, args, scales=4):
        super(MSCFB, self).__init__()
        G0 = args.n_feats
        kSize = args.kernel_size
        
        self.confusion_head = nn.Conv2d(G0, G0, 1, stride=1)
        self.conv_3_1 = nn.Conv2d(G0//scales, G0//scales, kSize, stride=1, padding=(kSize-1)//2)
        self.conv_3_2 = nn.Sequential(*[
            nn.Conv2d(G0//scales, G0//scales, kSize, stride=1, padding=(kSize-1)//2),
            nn.LeakyReLU(negative_slope = 0.05),
            nn.Conv2d(G0//scales, G0//scales, kSize, stride=1, padding=(kSize-1)//2)
        ])
        self.conv_3_3 = nn.Sequential(*[
            nn.Conv2d(G0//scales, G0//scales, kSize, stride=1, padding=(kSize-1)//2),
            nn.LeakyReLU(negative_slope = 0.05),
            nn.Conv2d(G0//scales, G0//scales, kSize, stride=1, padding=(kSize-1)//2),
            nn.LeakyReLU(negative_slope = 0.05),
            nn.Conv2d(G0//scales, G0//scales, kSize, stride=1, padding=(kSize-1)//2)
        ])
        self.lrelu = nn.LeakyReLU(negative_slope = 0.05)
        self.cca = CCALayer(args)
        self.confusion_tail = nn.Conv2d(G0, G0, 1,stride = 1)

    def forward(self, x):
        Low = self.confusion_head(x)
        X = torch.split(Low,8,dim=1)
        x1 = X[0]
        x2 = X[1]
        x3 = X[2]
        x4 = X[3]
        
        y1 = x1
        y2 = self.lrelu(self.conv_3_1(x2))
        y3 = self.lrelu(self.conv_3_2(x3))
        y4 = self.lrelu(self.conv_3_3(x4))

        out = torch.cat([y1,y2,y3,y4], 1)
        
        out = channel_shuffle(out, 4)
        out = self.confusion_tail(self.cca(out))
        return out + x


class LPNet(nn.Module):
    def __init__(self, args):
        super(LPNet, self).__init__()
        n_blocks = 4
        self.n_blocks = n_blocks
        in_channels = args.n_feats

        self.avg1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.avg2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.half_head = conv_layer(3, in_channels, 3)
        self.quater_head = conv_layer(3, in_channels, 3)

        self.half_fuse = conv_layer(in_channels*n_blocks,in_channels,1)
        self.quater_fuse = conv_layer(in_channels*n_blocks,in_channels,1)

        self.half_cat = conv_layer(in_channels*2,in_channels,1)

        self.half_tail = conv_layer(in_channels, in_channels, 3)
        self.quater_tail = conv_layer(in_channels, in_channels, 3)
        self.lrelu = nn.LeakyReLU()

        self.quater_up = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        self.half_up = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)

        half_body = nn.ModuleList()
        quater_body = nn.ModuleList()
        for i in range(n_blocks):
            half_body.append(
                MSCFB(args))
                #LightenBlock(args))

        for i in range(n_blocks):
            quater_body.append(
                MSCFB(args))
                #LightenBlock(args))


        self.half1 = nn.Sequential(*half_body)
        self.quater = nn.Sequential(*quater_body)

        self.head = conv_layer(3, in_channels, 3)
        # define body module
        modules_body = nn.ModuleList()
        for i in range(n_blocks):
            modules_body.append(
                MSCFB(args))

        lighten_block = nn.ModuleList()
        for i in range(n_blocks):
            lighten_block.append(
                MSCFB(args))
                #LightenBlock(args))

        self.body = nn.Sequential(*modules_body)
        self.lighten = nn.Sequential(*lighten_block)
        self.cat1 = conv_layer(in_channels*2,in_channels,1)
        self.conv1 = conv_layer(in_channels*n_blocks,in_channels,1)
        self.conv3 = conv_layer(in_channels,in_channels,3)
        self.tail = conv_layer(in_channels,3,3)
        self.transform = conv_layer(in_channels,3,3)
    
    def forward(self, x):
        half_img = self.avg1(x)
        quater_img = self.avg2(half_img)
        quaterhead = self.quater_head(quater_img)
        

        half_out = []
        quater_out = []
        for i in range(self.n_blocks):
            quaterhead = self.quater[i](quaterhead)
            quater_out.append(quaterhead)

        quaterfuse = self.lrelu(self.quater_fuse(torch.cat(quater_out,1)))
        quatertail = self.lrelu(self.quater_tail(quaterfuse))
        quaterup = self.quater_up(quatertail)
        
        
        halfhead = self.half_head(half_img)
        quaterup = F.interpolate(quaterup, size=halfhead.size()[2:], scale_factor=None, mode='bilinear', align_corners=True)
        #quaterup = F.upsample(quaterup, halfhead.size()[2:], mode='bilinear')
        halfhead = self.half_cat(torch.cat([halfhead,quaterup], dim=1))

        for i in range(self.n_blocks):
            halfhead = self.half1[i](halfhead)
            half_out.append(halfhead)

        halffuse = self.lrelu(self.half_fuse(torch.cat(half_out,1)))
        halftail = self.lrelu(self.half_tail(halffuse))
        halfup = self.half_up(halftail)

        x = self.head(x)
        res = x
        halfup = F.interpolate(halfup, size=x.size()[2:], scale_factor=None, mode='bilinear', align_corners=True)
        x = self.lrelu(self.cat1(torch.cat([x,halfup],dim=1)))
        y = x

        MSCFB_out = []
        lighten_out = []
        
        for i in range(self.n_blocks):
            x = self.body[i](x)
            
            y = self.lighten[i](y)
            
            #introduce position attention 
            #x = self.pam(x,y)
            x = x + y
            lighten_out.append(y)
            MSCFB_out.append(x)        

        rls = torch.cat(MSCFB_out,1)
        fuse = self.lrelu(self.conv1(rls))
        conv3 = self.lrelu(self.conv3(fuse))
        out = self.tail(conv3+res)

        out0 = self.transform(lighten_out[0])
        out1 = self.transform(lighten_out[1])
        out2 = self.transform(lighten_out[2])
        out3 = self.transform(lighten_out[3])

        output_all = [out, out0, out1, out2, out3]
        #output_all = [out, out0, out1,out2]

        return output_all



