import torch
from deeplab2s.base import initialization as init


class SegmentationModel(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.sssr_decoder)
        init.initialize_decoder(self.sisr_decoder)
        init.initialize_head(self.segmentation_head)
        init.initialize_head(self.sisr_LR)
    
    def forward(self, x):
        """Sequentially pass 'x' trough model's encoder, ASPP, decoder, segmentation head and extra SR module"""
        
        ###############################################################################################################################
        ########################################## SHARED ENCODER #####################################################################
        ###############################################################################################################################
        
        features_resnet = self.encoder(x) # list of features at all levels [F0, F1, F2, F3, F4, F5]
        
        # For ResNet 101:   F0 (Bs, 3, H, W)
        #                   F1 (Bs, 64, H/2, W/2)
        #                   F2 (Bs, 256, H/4, W/4)
        #                   F3 (Bs, 512, H/8, W/8)
        #                   F4 (Bs, 1024, H/16, W/16)
        #                   F5 (Bs, 2048, H/16, W/16), F5 is dilated with rate 2 so the spatial dimensions are not reduced
        
        #print(f'features_resnet[-1], F5: {features_resnet[-1].shape}') # Should be (Bs, 2048, H/16, W/16)
        
        # F5 enters the ASPP module, which applies atrous convolutions at multiple rates (12, 24, 36)
        #
        # The ASPP module does in parallel:    -1x1 conv (256 channels)
        #                                      -3x3 conv, rate 12 (256 channels)
        #                                      -3x3 conv, rate 24 (256 channels)
        #                                      -3x3 conv, rate 36 (256 channels)
        #                                      -Image Pooling (256 channels)
        #
        # Moreover, it also does a 1x1 conv to fuse the 5 * 256 channels to just 128 channels
                            
        aspp_features = self.aspp(features_resnet[-1]) # The output has shape (Bs, 128, H/16, W/16)
        
        # This point is considered the output of the encoder
        
        #print(f'aspp_features, i.e., encoder output: {aspp_features.shape}') # Should be (Bs, 128, H/16, W/16)
        
        #print()
        ###############################################################################################################################
        ############################################## SSSR DECODER ###################################################################
        ###############################################################################################################################
        
        # The Decoder does the following:     -Upsample encoder output x4 -> shape (Bs, 128, H/4, W/4)
        #                                     -Concatenate with low-level features F2 (firstly reduced to just 48 channels) -> shape (Bs, 128+48, H/4, W/4)
        #                                     -3x3 conv, 128 filters to fuse the feature maps -> shape (Bs, 128, H/4, W/4)
        #                                     -3x3 conv, 128 filters
        #                                     -Upsample x2 -> shape (Bs, 128, H/2, W/2)
        #                                     -Concatenate with low-level features F1 -> shape (Bs, 128+64, H/2, W/2)
        #                                     -3x3 conv, 128 filters to fuse the feature maps -> shape (Bs, 128, H/2, W/2)
        
        sssr_decoder_out = self.sssr_decoder(aspp_features, *features_resnet)
        
        #print(f'sssr_decoder_out: {sssr_decoder_out.shape}') # Should be (Bs, 128, H/2, W/2)
        
        # The Segmentation Head does the following:     -1x1 conv, num_classes filters to reduce the number of channels -> shape (Bs, num_classes, H/2, W/2)
        #                                               -Upsample x2 -> shape (Bs, num_classes, H, W)
        
        sssr_LR = self.segmentation_head(sssr_decoder_out)
        
        #print(f'sssr_LR: {sssr_LR.shape}') # Should be (Bs, num_classes, H, W)
        
        # The extra up-sampling module does the following:      -3x3 conv, num_classes filters
        #                                                       -Upsample x2 -> shape (Bs, num_classes, 2*H, 2*W)
        #                                                       -3x3 conv, num_classes filters
        
        sssr_HR = self.sssr(sssr_LR) 
        
        #print(f'sssr_HR: {sssr_HR.shape}') # Should be (Bs, num_classes, 2*H, 2*W)
        
        #print()
        ################################################################################################################################
        ############################################# SISR DECODER #####################################################################
        ################################################################################################################################
        
        # The Decoder does the following:     -Upsample encoder output x4 -> shape (Bs, 128, H/4, W/4)
        #                                     -Concatenate with low-level features F2 (firstly reduced to just 48 channels) -> shape (Bs, 128+48, H/4, W/4)
        #                                     -3x3 conv, 128 filters to fuse the feature maps -> shape (Bs, 128, H/4, W/4)
        #                                     -3x3 conv, 128 filters
        #                                     -Upsample x2 -> shape (Bs, 128, H/2, W/2)
        #                                     -Concatenate with low-level features F1 -> shape (Bs, 128+64, H/2, W/2)
        #                                     -3x3 conv, 128 filters to fuse the feature maps -> shape (Bs, 128, H/2, W/2)
        
        sisr_decoder_out = self.sisr_decoder(aspp_features, *features_resnet)
        
        #print(f'sisr_decoder_out: {sisr_decoder_out.shape}') # Should be (Bs, 128, H/2, W/2)
        
        # The "Segmentation Head" does the following:     -1x1 conv, num_classes filters to reduce the number of channels -> shape (Bs, 64, H/2, W/2)
        #                                                 -Upsample x2 -> shape (Bs, 64, H, W)
        
        sisr_LR = self.sisr_LR(sisr_decoder_out)
        
        #print(f'sisr_LR: {sisr_LR.shape}') # Should be (Bs, 64, H, W)
        
        # The extra up-sampling module does the following:      -3x3 conv, 3 filters to reduce from 64 to 3 channels
        #                                                       -Upsample x2 -> shape (Bs, 3, 2*H, 2*W)
        #                                                       -3x3 conv, 3 filters
        
        sisr_HR = self.sisr(sisr_LR)
        
        #print(f'sisr_HR: {sisr_HR.shape}') # Should be (Bs, 3, 2*H, 2*W)
        
        #print()
        ################################################################################################################################
        ################################################ FA MODULE #####################################################################
        ################################################################################################################################
        
        # The Feature Affinity does the following:      -1x1 conv to reduce the sssr_HR from num_classes to just 3 channels in order to compute the similarity matrix
        
        sssr_FA = self.sssr_FA(sssr_HR)
        
        #print(f'sssr_FA: {sssr_FA.shape}') # Should be (Bs, 3, 2*H, 2*W)
        
        return (sssr_HR, sisr_HR, sssr_FA)

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`
        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)
        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)
        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x