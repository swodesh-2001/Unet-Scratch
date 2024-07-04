import torch
import torch.nn as nn 

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size= 3,padding= 1 , stride= 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels,out_channels,kernel_size= 3,padding= 1 , stride= 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
        )
    
    def forward(self, x):
        return self.conv(x)


class Downs(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down_conv = self.conv(x)
        down_pool = self.pool(down_conv)

        return down_conv, down_pool
    

class Ups(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
       x1 = self.up(x1)
       x = torch.cat([x1, x2], 1)
       return self.conv(x)
    

class UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_convolution_1 = Downs(in_channels, 64)
        self.down_convolution_2 = Downs(64, 128)
        self.down_convolution_3 = Downs(128, 256)
        self.down_convolution_4 = Downs(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up_convolution_1 = Ups(1024, 512)
        self.up_convolution_2 = Ups(512, 256)
        self.up_convolution_3 = Ups(256, 128)
        self.up_convolution_4 = Ups(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels= out_channels, kernel_size=1)

    def forward(self, x):
       down_1, pool_1 = self.down_convolution_1(x)
       down_2, pool_2 = self.down_convolution_2(pool_1)
       down_3, pool_3 = self.down_convolution_3(pool_2)
       down_4, pool_4 = self.down_convolution_4(pool_3)

       bottom_conv = self.bottle_neck(pool_4)

       up_1 = self.up_convolution_1(bottom_conv, down_4)
       up_2 = self.up_convolution_2(up_1, down_3)
       up_3 = self.up_convolution_3(up_2, down_2)
       up_4 = self.up_convolution_4(up_3, down_1)

       out = self.out(up_4)
       return out






 
