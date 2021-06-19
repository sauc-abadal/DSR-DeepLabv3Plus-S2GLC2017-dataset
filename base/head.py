import torch.nn as nn

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, upsampling, upsampling_type='Upsample'):
        super().__init__()
        
        if upsampling_type == 'Upsample':
            self.up = nn.Upsample(scale_factor=upsampling, mode='nearest')
        elif upsampling_type == 'TransposeConv': 
            self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2, padding=0)
        elif upsampling_type == 'PixelShuffle': 
            self.up = Pixel_Shuffle_Module(in_channels, in_channels, upsampling)
        else:
            self.up = nn.Upsample(scale_factor=upsampling, mode='nearest')
        
        self.conv = nn.Sequential(
               nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
    
    def forward(self, x):        
        x = self.up(x)
        x = self.conv(x)
        
        return x
    
class Pixel_Shuffle_Module(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels*(scale_factor**2), kernel_size=3, stride=1, padding=1)

        self.up = nn.PixelShuffle(scale_factor)
    
    def forward(self, LR_in):
        
        HR_out = self.conv(LR_in)
        
        HR_out = self.up(HR_out)
        
        return HR_out