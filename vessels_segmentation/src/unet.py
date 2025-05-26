#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ashvaro
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
#from utils_pytorch import SaveFeatureMaps
import math
from collections import OrderedDict


class ConvBlock(nn.Module):
    
    def __init__(self, input_ch=3, output_ch=64, activf=nn.ReLU, bias=True):
        super(ConvBlock, self).__init__()
            
        conv1 = nn.Conv2d(input_ch, output_ch, 3, 1, 1, bias=bias)
        conv2 = nn.Conv2d(output_ch, output_ch, 3, 1, 1, bias=bias)
      
        self.conv_block = nn.Sequential(conv1,
                                        activf(inplace=True),
                                        conv2,
                                        activf(inplace=True))
        
    def forward(self, x):
        return self.conv_block(x)
        
    

class UpConv(nn.Module):
    
    def __init__(self, input_ch=64, output_ch=32, activf=nn.ReLU, bias=True):
        super(UpConv, self).__init__()
            
        conv = nn.ConvTranspose2d(input_ch, output_ch, 2, 2, bias=bias)
#        self.activf = activf
        
        self.conv_block = nn.Sequential(conv)
#                                        self.activf(inplace=True))
        
    def forward(self, x):
        return self.conv_block(x)
   
    

class DownConv(nn.Module):
    
    def __init__(self, input_ch=64, output_ch=32, activf=nn.ReLU, bias=True):
        super(DownConv, self).__init__()
            
        conv = nn.Conv2d(input_ch, output_ch, 4, 2, bias=bias)
#        self.activf = activf
        
        self.conv_block = nn.Sequential(conv)
#                                        self.activf(inplace=True))
        
    def forward(self, x):
        return self.conv_block(x)



class UNet(nn.Module):
    
    def __init__(self, input_ch, output_ch, base_ch):
        super(UNet, self).__init__()
        
        
        self.conv1 = ConvBlock(input_ch, base_ch)
        self.conv2 = ConvBlock(base_ch, 2*base_ch)
        self.conv3 = ConvBlock(2*base_ch, 4*base_ch)
        self.conv4 = ConvBlock(4*base_ch, 8*base_ch)
        self.conv5 = ConvBlock(8*base_ch, 16*base_ch)
        
        self.upconv1 = UpConv(16*base_ch,8*base_ch)
        self.conv6 = ConvBlock(16*base_ch, 8*base_ch)   
        self.upconv2 = UpConv(8*base_ch,4*base_ch)
        self.conv7 = ConvBlock(8*base_ch, 4*base_ch)
        self.upconv3 = UpConv(4*base_ch,2*base_ch)
        self.conv8 = ConvBlock(4*base_ch, 2*base_ch)       
        self.upconv4 = UpConv(2*base_ch,base_ch)
        self.conv9 = ConvBlock(2*base_ch, base_ch)  
        
        self.outconv = nn.Conv2d(base_ch,output_ch,1,bias=True)  
        
        
    def forward(self, x):

        x1 = self.conv1(x)
        x= F.max_pool2d(x1,2,2)

        x2 = self.conv2(x)
        x= F.max_pool2d(x2,2,2)
        
        x3 = self.conv3(x)
        x= F.max_pool2d(x3,2,2)
        
        x4 = self.conv4(x)
        x= F.max_pool2d(x4,2,2)
        
        x = self.conv5(x)
        x = self.upconv1(x)
        x = torch.cat((x4,x),dim=1)
        
        x = self.conv6(x)
        x = self.upconv2(x)
        x = torch.cat((x3,x),dim=1)

        x = self.conv7(x)
        x = self.upconv3(x)
        x = torch.cat((x2,x),dim=1)

        x = self.conv8(x)
        x = self.upconv4(x)
        x = torch.cat((x1,x),dim=1)

        x = self.conv9(x)
        x = self.outconv(x)

        return x #, internal           


    def weight_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.kaiming_normal(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant(m.bias.data, 0)
                print(m)
        if isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal(m.weight.data, a=0, mode='fan_out', nonlinearity='conv_transpose2d')
            if m.bias is not None:
                init.constant(m.bias.data, 0)
                print(m)
        if isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight.data, 1)
            init.constant(m.bias.data, 0)
            print(m)


    def initialize(self):
        self.apply(self.weight_init)
        init.kaiming_normal(self.outconv.weight.data, a=0, mode='fan_out', nonlinearity='linear')
        init.constant(self.outconv.bias.data, 0)
        
    def set_eval(self):
        pass
#        self.dropout1.eval()
#        self.dropout2.eval()
#        self.dropout3.eval()
#        self.dropout4.eval()
#        
    def set_train(self):
        pass
#        self.dropout1.train(True)
#        self.dropout2.train(True)
#        self.dropout3.train(True)
#        self.dropout4.train(True)     



class UNetDrop(nn.Module):
    
    def __init__(self, input_ch, output_ch, base_ch, drop_p=0.2):
        super(UNetDrop, self).__init__()
                
        self.conv1 = ConvBlock(input_ch, base_ch)
        self.conv2 = ConvBlock(base_ch, 2*base_ch)
        self.conv3 = ConvBlock(2*base_ch, 4*base_ch)
        self.conv4 = ConvBlock(4*base_ch, 8*base_ch)
        self.conv5 = ConvBlock(8*base_ch, 16*base_ch)
        
        self.upconv1 = UpConv(16*base_ch,8*base_ch)
        self.conv6 = ConvBlock(16*base_ch, 8*base_ch)   
        self.upconv2 = UpConv(8*base_ch,4*base_ch)
        self.conv7 = ConvBlock(8*base_ch, 4*base_ch)
        self.upconv3 = UpConv(4*base_ch,2*base_ch)
        self.conv8 = ConvBlock(4*base_ch, 2*base_ch)       
        self.upconv4 = UpConv(2*base_ch,base_ch)
        self.conv9 = ConvBlock(2*base_ch, base_ch)  
        
        self.outconv = nn.Conv2d(base_ch,output_ch,1,bias=True) 
        
        self.dropout2 = nn.Dropout(drop_p)
        self.dropout3 = nn.Dropout(drop_p)
        self.dropout4 = nn.Dropout(drop_p)
        self.dropout5 = nn.Dropout(drop_p)
        self.dropout6 = nn.Dropout(drop_p)
        self.dropout7 = nn.Dropout(drop_p)        
     
    def forward(self, x):

        x1 = self.conv1(x)
        x= F.max_pool2d(x1,2,2)

        x2 = self.conv2(x)
        x= F.max_pool2d(x2,2,2)
        x = self.dropout2(x)
        
        x3 = self.conv3(x)
        x= F.max_pool2d(x3,2,2)
        x = self.dropout3(x)
        
        x4 = self.conv4(x)
        x= F.max_pool2d(x4,2,2)
        x = self.dropout4(x)
        
        x = self.conv5(x)
        x = self.dropout5(x)
        x = self.upconv1(x)
        x = torch.cat((x4,x),dim=1)
        
        x = self.conv6(x)
        x = self.dropout6(x)
        x = self.upconv2(x)
        x = torch.cat((x3,x),dim=1)

        x = self.conv7(x)
        x = self.dropout7(x)
        x = self.upconv3(x)
        x = torch.cat((x2,x),dim=1)

        x = self.conv8(x)
        x = self.upconv4(x)
        x = torch.cat((x1,x),dim=1)

        x = self.conv9(x)
        x = self.outconv(x)

        return x #, internal           


    def weight_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.kaiming_normal(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant(m.bias.data, 0)
                print(m)
        if isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal(m.weight.data, a=0, mode='fan_out', nonlinearity='conv_transpose2d')
            if m.bias is not None:
                init.constant(m.bias.data, 0)
                print(m)
        if isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight.data, 1)
            init.constant(m.bias.data, 0)
            print(m)


    def initialize(self):
        self.apply(self.weight_init)
        init.kaiming_normal(self.outconv.weight.data, a=0, mode='fan_out', nonlinearity='linear')
        init.constant(self.outconv.bias.data, 0)
        
    def set_eval(self):
        self.dropout2.eval()
        self.dropout3.eval()
        self.dropout4.eval()
        self.dropout5.eval()
        self.dropout6.eval()
        self.dropout7.eval()
#        
    def set_train(self):
        self.dropout2.train(True)
        self.dropout3.train(True)
        self.dropout4.train(True) 
        self.dropout5.train(True)
        self.dropout6.train(True)
        self.dropout7.train(True)        


