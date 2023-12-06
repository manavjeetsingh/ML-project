import torch.nn as nn
import torch
import torchvision.models
import torch.nn.functional as F
# from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet34, ResNet34_Weights


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None,dilation1=1,dilation2=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False,dilation=dilation1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False,dilation=dilation2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,dilation1=1,dilation2=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels,dilation1=dilation1,dilation2=dilation2)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# class DownAtrous(nn.Module):
#     """Downscaling with maxpool then double conv"""

#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             DoubleConv(in_channels, out_channels,dilation=2)
#         )

#     def forward(self, x):
#         return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)





class UNet(nn.Module):
    def __init__(self, n_classes, bilinear=False,load_pretrained_encoder_layers=False):
        super(UNet, self).__init__()
        self.n_channels = 3
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_layers = []

        self.inc = (DoubleConv(3, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits





class Dilation1(nn.Module):
    def __init__(self, n_classes, bilinear=False,load_pretrained_encoder_layers=False):
        super().__init__()
        self.n_channels = 3
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_layers = []

        self.inc = (DoubleConv(3, 64,dilation1=1,dilation2=1))
        self.down1 = (Down(64, 128,dilation1=2,dilation2=1))
        self.down2 = (Down(128, 256,dilation1=2,dilation2=1))
        self.down3 = (Down(256, 512,dilation1=2,dilation2=1))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class Dilation2(nn.Module):
    def __init__(self, n_classes, bilinear=False,load_pretrained_encoder_layers=False):
        super().__init__()
        self.n_channels = 3
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_layers = []

        self.inc = (DoubleConv(3, 64,dilation1=1,dilation2=1))
        self.down1 = (Down(64, 128,dilation1=2,dilation2=2))
        self.down2 = (Down(128, 256,dilation1=2,dilation2=2))
        self.down3 = (Down(256, 512,dilation1=2,dilation2=2))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    


class UNet2Layer(nn.Module):
    def __init__(self, n_classes, bilinear=False,load_pretrained_encoder_layers=False):
        print("--------------2layers------------")
        super().__init__()
        self.n_channels = 3
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_layers = []

        self.inc = (DoubleConv(3, 64))
        self.down1 = (Down(64, 128))
        factor = 2 if bilinear else 1
        # self.down2 = (Down(128, 256))
        # self.down3 = (Down(256, 512))
        self.down2 = (Down(128, 256 // factor))
        # self.up1 = (Up(1024, 512 // factor, bilinear))
        # self.up2 = (Up(512, 256 // factor, bilinear))
        self.up1 = (Up(256, 128 // factor, bilinear))
        self.up2 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    

class UNet2LayerDilation(nn.Module):
    def __init__(self, n_classes, bilinear=False,load_pretrained_encoder_layers=False):
        print("--------------2layers------------")
        super().__init__()
        self.n_channels = 3
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_layers = []

        self.inc = (DoubleConv(3, 64))
        self.down1 = (Down(64, 128,dilation1=2,dilation2=2))
        factor = 2 if bilinear else 1
        # self.down2 = (Down(128, 256))
        # self.down3 = (Down(256, 512))
        self.down2 = (Down(128, 256 // factor,dilation1=2,dilation2=2))
        # self.up1 = (Up(1024, 512 // factor, bilinear))
        # self.up2 = (Up(512, 256 // factor, bilinear))
        self.up1 = (Up(256, 128 // factor, bilinear))
        self.up2 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_ReLU=True):
        super().__init__()
        self.use_ReLU=use_ReLU
        
        if use_ReLU:
            self.layers= nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        else: 
            self.layers= nn.Sequential(
                nn.Conv2d(in_ch,out_ch,kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        
    def forward(self, x):
        return self.layers(x)





# class Residual2LayerUNet(nn.Module):
#     def __init__(self, n_classes, load_pretrained_encoder_layers=False):
#         super().__init__()

#         if load_pretrained_encoder_layers:
#             resnet = resnet34(weights=ResNet34_Weights.DEFAULT)
#             l1=resnet.layer1
#             l2=resnet.layer2

#             for param in l1.parameters():
#                 param.requires_grad = False

#             for param in l2.parameters():
#                 param.requires_grad = False
#         else:
#             resnet=resnet34(weights=None)
#             l1=resnet.layer1
#             l2=resnet.layer2
        
#         self.resnet=resnet
#         self.base_layers = list(self.resnet.children())

#         ## 0. Pre layer
#         self.pre_layer=nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

#         ## 1. Encoder containing layers of Resnet
#         self.encoder_l1 = l1
        
#         self.encoder_l2 = l2
#         # self.encoder_l3 = resnet.layer3
#         # self.encoder_l4 = resnet.layer4


#         # 2. Decoder layers
#         x, y = 10, 10
#         # self.decoder_l1 = DecoderBlock(512,256//2)
#         # self.decoder_l2 = DecoderBlock(384,128//2)
#         self.decoder_l3 = DecoderBlock(128,64//2)
#         self.decoder_l4 = DecoderBlock(96,64//2)

#         # 3. Out layer
#         self.out_layer = DecoderBlock(35,n_classes,use_ReLU=False)

        
#     def forward(self, iput):
#         # print("iput:",iput.shape)
#         preout=self.pre_layer(iput)
#         # print("preout:",preout.shape)
#         eout1=self.encoder_l1(preout)
#         # print("eout1:",eout1.shape)
#         eout2=self.encoder_l2(eout1)
#         # print("eout2:",eout2.shape)
#         # eout3=self.encoder_l3(eout2)
#         # # print("eout3:",eout3.shape)
#         # eout4=self.encoder_l4(eout3)
#         # print("eout4:",eout4.shape)

#         # dout1=self.decoder_l1(eout4)
#         # dout1=torch.cat([dout1,eout3],1)
#         # # print("dout1:",dout1.shape)

#         # dout2=self.decoder_l2(dout1)
#         # dout2=torch.cat([dout2,eout2],1)
#         # print("dout2:",dout2.shape)

#         dout3=self.decoder_l3(eout2)
#         dout3=torch.cat([dout3,eout1],1)
#         # print("dout3:",dout3.shape)

#         dout4=self.decoder_l4(dout3)
#         dout4=torch.cat([dout4,iput],1)
#         # print("dout4:",dout4.shape)

#         out=self.out_layer(dout4)
#         # print("out:",out.shape)
# # 
#         return out


class Residual2LayerUNet(nn.Module):
    def __init__(self, n_classes, bilinear=False,load_pretrained_encoder_layers=False):
        print("--------------------------------------------------PRE LOADED",load_pretrained_encoder_layers)

        if load_pretrained_encoder_layers:
            resnet = resnet34(weights=ResNet34_Weights.DEFAULT)
            l1=resnet.layer1
            l2=resnet.layer2

            for param in l1.parameters():
                param.requires_grad = False

            for param in l2.parameters():
                param.requires_grad = False
        else:
            resnet=resnet34(weights=None)
            l1=resnet.layer1
            l2=resnet.layer2


        super().__init__()
        self.n_channels = 3
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_layers = []

        self.inc = (DoubleConv(3, 64))
        self.down1 = l1
        factor = 2 if bilinear else 1
        self.down2 = l2
        self.up1 = (Up(128, 128 // factor, bilinear))
        self.up2 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        # print("x1",x1.shape)
        x2 = self.down1(x1)
        # print("x2",x2.shape)
        x3 = self.down2(x2)
        # print("x3",x3.shape)
        x = self.up1(x3, x2)
        # print("x",x.shape)
        x = self.up2(x, x1)
        # print("x",x.shape)
        logits = self.outc(x)
        # print("logits",logits.shape)
        return logits