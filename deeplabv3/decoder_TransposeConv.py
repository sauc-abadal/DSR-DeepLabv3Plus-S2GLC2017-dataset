import torch
from torch import nn
from torch.nn import functional as F


class DeepLabV3PlusDecoder(nn.Module):
    def __init__(self, encoder_channels, out_channels=128, output_stride=16):
        super().__init__()
        
        if output_stride not in {8, 16}:
            raise ValueError("Output stride should be 8 or 16, got {}.".format(output_stride))

        self.out_channels = out_channels
        self.output_stride = output_stride

        scale_factor = 2 if output_stride == 8 else 4
        # Upsampling of the "green" feature maps in the paper
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=2, stride=2, padding=0),
            SeparableConv2d(
                self.out_channels,
                self.out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.ConvTranspose2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=2, stride=2, padding=0)
        )            
            
        highres_in_channels = encoder_channels[-4]
        highres_out_channels = 48   # proposed by authors of paper
        
        highres_out_channels2 = encoder_channels[-5] 
        
        # 1x1 Conv to reduce the number of channels of high resolution feature maps (F2) used in the decoder
        self.block1 = nn.Sequential(
            nn.Conv2d(highres_in_channels, highres_out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(highres_out_channels),
            nn.ReLU(),
        )
        
        # 3x3 Conv to fuse the high resolution features (F2) with the ASPP features
        self.block2 = nn.Sequential(
            SeparableConv2d(
                highres_out_channels + self.out_channels,
                self.out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
        )
        # Added the other 3x3 Conv, 128 filters
        self.block3 = nn.Sequential(
            SeparableConv2d(
                self.out_channels,
                self.out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
        )
        
        self.up2 = nn.ConvTranspose2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=2, stride=2, padding=0)
        
        self.block4 = nn.Sequential(
            SeparableConv2d(
                highres_out_channels2 + self.out_channels,
                self.out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
        )
        # The final upsampling by 2 is done in the SegmentationHead part, with the projection of the number of channels to the desired classes
        # Another upsampling by 2 is done in the final upsampling module for Super-Resolution
        
    def forward(self, aspp_features, *features_resnet):
        # Upsample the ASPP features (output of the encoder) x4, 128 channels
        aspp_features = self.up(aspp_features)

        # Reduce the low-level resnet features (F2) channels from 256 to 48
        high_res_features = self.block1(features_resnet[-4])
        
        # Concatenate both feature maps (First skip connection) -> 128 + 48 = 176 channels
        concat_features = torch.cat([aspp_features, high_res_features], dim=1)
        
        # Conv 3x3, 128 filters to fuse the feature maps to 128 channels
        fused_features = self.block2(concat_features)
        
        # 2nd Conv 3x3, 128 filters
        fused_features = self.block3(fused_features)
        
        # Upsample x2
        fused_features = self.up2(fused_features)
        
        # Concatenate with F1 feature maps (Long skip connection) -> 128 + 64 = 192 channels
        concat_features2 = torch.cat([fused_features, features_resnet[-5]], dim=1)
        
        # Conv 3x3, 128 filters to fuse the feature maps to 128 channels
        fused_features = self.block4(concat_features2)
        return fused_features



class SeparableConv2d(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        dephtwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        super().__init__(dephtwise_conv, pointwise_conv)
