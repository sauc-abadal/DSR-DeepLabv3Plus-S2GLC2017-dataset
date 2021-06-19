import torch.nn as nn

from typing import Optional

from .decoder import DeepLabV3PlusDecoder

from base import SegmentationModel
from base import SegmentationHead
from encoders import get_encoder

from torch.nn import functional as F
import torch

class DeepLabV3Plus(SegmentationModel):
    """DeepLabV3+ implementation from "Encoder-Decoder with Atrous Separable
    Convolution for Semantic Image Segmentation"
    
    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features 
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and 
            other pretrained weights (see table with available weights for each encoder_name)
        encoder_output_stride: Downsampling factor for last encoder features (see original paper for explanation)
        decoder_atrous_rates: Dilation rates for ASPP module (should be a tuple of 3 integer values)
        decoder_channels: A number of convolution filters in ASPP module. Default is 256
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
            Default is **None**
        upsampling: Final upsampling factor. Default is 4 to preserve input-output spatial shape identity
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build 
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax" (could be **None** to return logits)
    Returns:
        ``torch.nn.Module``: **DeepLabV3Plus**
    
    Reference:
        https://arxiv.org/abs/1802.02611v3

    """
    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            encoder_output_stride: int = 16,
            decoder_channels: int = 128,
            decoder_atrous_rates: tuple = (12, 24, 36),
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[str] = None,
            upsampling: int = 2,
            aux_params: Optional[dict] = None,
    ):
        super().__init__()
        
       ###############################################################################################################################
        ########################################## SHARED ENCODER #####################################################################
        ###############################################################################################################################
        
        self.encoder = get_encoder(encoder_name, in_channels=in_channels, depth=encoder_depth, weights=encoder_weights)

        if encoder_output_stride == 8:
            self.encoder.make_dilated(stage_list=[4, 5], dilation_list=[2, 4])
            
        elif encoder_output_stride == 16:
            self.encoder.make_dilated(stage_list=[5], dilation_list=[2]) # Normally OS will be 16 in order to achieve better computation efficiency,
                                                                         # only the last stage (depth 5) is dilated with rate = 2
            
        else:
            raise ValueError("Encoder output stride should be 8 or 16, got {}".format(encoder_output_stride))
            
        self.aspp = nn.Sequential(
            ASPP(self.encoder.out_channels[-1], decoder_channels, 128, decoder_atrous_rates, separable=True) #output of "green" feature map in the paper
        )
        
        ###############################################################################################################################
        ############################################## SSSR DECODER ###################################################################
        ###############################################################################################################################
            
        self.sssr_decoder = DeepLabV3PlusDecoder(
            encoder_channels=self.encoder.out_channels, 
            out_channels=decoder_channels, 
            output_stride=encoder_output_stride)

        # prova, SegmentationHead manté els 128 canals -> SSSR extra UP fa l'upsampling als Features i dsprs passa a classes
        self.segmentation_head = SegmentationHead(
            in_channels=self.sssr_decoder.out_channels, 
            out_channels=self.sssr_decoder.out_channels, 
            activation=activation, 
            kernel_size=1, 
            upsampling=upsampling)
        
        self.sssr = sssr_extra_upsampling_module(in_ch=self.sssr_decoder.out_channels, out_ch=classes)
        
        ################################################################################################################################
        ############################################# SISR DECODER #####################################################################
        ################################################################################################################################
            
        self.sisr_decoder = DeepLabV3PlusDecoder(
            encoder_channels=self.encoder.out_channels, 
            out_channels=decoder_channels, 
            output_stride=encoder_output_stride)

        self.sisr_LR = SegmentationHead(
            in_channels=self.sisr_decoder.out_channels, 
            out_channels=64, 
            activation=activation, 
            kernel_size=1, 
            upsampling=upsampling)
        
        self.sisr = sisr_extra_upsampling_module(in_ch=64, out_ch=4)
        
        ################################################################################################################################
        ################################################ FA MODULE #####################################################################
        ################################################################################################################################
            
        self.sssr_FA = sssr_feature_transf(classes, 4)
        
###########################################################################################################################################################     
########################## ASPP Module definition and ASPP function #######################################################################################
###########################################################################################################################################################

class SeparableConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        dephtwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=False)
        pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        super().__init__(dephtwise_conv, pointwise_conv)

# Class used for computing 3x3 Conv at a given rate of the ASPP
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
# Same, but in a separate manner, i.e. SeparableConv2d instead of Conv2d        
class ASPPSeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            SeparableConv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

# Class used for computing the Image Pooling branch of the ASPP
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1), # Image pooling in the spatial dimensions to a fixed size 1x1, channels remain the same
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), # 1x1 Conv to project to just 256 channels
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[-2:] # spatial dimensions
        
        # pass the feature map through the nn modules sequence
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False) # upsamples /downsamples the feature map to the same shape as before pooling


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, out_channels2, atrous_rates, separable=False):
        super(ASPP, self).__init__()
        
        modules = [] # list of all the modules of the ASPP, i.e. 1x1 Conv, 3x3 Conv rate 12, 3x3 Conv rate 24, 3x3 Conv rate 36 and Image Pooling
        
        # 1x1 Conv
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        ASPPConvModule = ASPPConv if not separable else ASPPSeparableConv
        
        # 3x3 Conv rate 12
        modules.append(ASPPConvModule(in_channels, out_channels, rate1))
        
        # 3x3 Conv rate 24
        modules.append(ASPPConvModule(in_channels, out_channels, rate2))
        
        # 3x3 Conv rate 36
        modules.append(ASPPConvModule(in_channels, out_channels, rate3))
        
        # Image Pooling
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        
        # Conv 1x1 that projects all the feature maps of the ASPP (5*256 channels) to just 128 channels
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels2),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res) # ENCODER OUTPUT, i.e. self.aspp output (BS, 256, H/16, W/16)
    
###############################################################################################################

def sssr_extra_upsampling_module(in_ch, out_ch):
    out = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
    )
    return out

def sisr_extra_upsampling_module(in_ch, out_ch):
    out = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
    )
    return out

def sssr_feature_transf(inp_ch, out_ch):
    out = nn.Sequential(
        nn.Conv2d(in_channels=inp_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0),
    )
    return out
    