import torch.nn as nn

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, upsampling, upsampling_type='Upsample'):
        super().__init__()
        
        if upsampling_type == 'Upsample':
            self.up = nn.Upsample(scale_factor=upsampling, mode='nearest')
        elif upsampling_type == 'TransposeConv': 
            self.up = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0)
        else:
            self.up = nn.Upsample(scale_factor=upsampling, mode='nearest')
        
        self.conv = nn.Sequential(
               nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
    
    def forward(self, x):        
        x = self.up(x)
        x = self.conv(x)
        
        return x