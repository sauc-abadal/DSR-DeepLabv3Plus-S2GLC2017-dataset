import torch
import torch.utils.data
import numpy as np
import os
from skimage.transform import resize

class DatasetCorine(torch.utils.data.Dataset):
    def __init__(self, corine_images_path, corine_labels_path, augmentation=None):

        self.img_h = 1000
        self.img_w = 1000

        self.new_img_h = 256
        self.new_img_w = 256
        
        self.ids = sorted(os.listdir(corine_images_path)) #returns image names
        self.images = [os.path.join(corine_images_path, image_id) for image_id in self.ids]
        self.labels = [os.path.join(corine_labels_path, label_id) for label_id in self.ids]
        self.augmentation = augmentation

        self.num_examples = len(self.ids)

    def __getitem__(self, index):
        
        gt_sisr = np.load(self.images[index]) # (shape: (4, 1000, 1000))
        gt_sssr = np.load(self.labels[index]) # (shape: (1000, 1000))
        
        if self.augmentation:
            sample = self.augmentation(image=gt_sisr, mask=gt_sssr)
            gt_sisr, gt_sssr = sample['image'], sample['mask']
        
        gt_sisr = np.transpose(gt_sisr, (1, 2, 0)) # (shape: (1000, 1000, 4))
        
        ################################################################################################################
        # select a 512x512 random crop from the img and labels (gt_sssr and gt_sisr) and downsample the img to 256x256:
        ################################################################################################################
        start_x = np.random.randint(low=0, high=(self.img_w - 512))
        end_x = start_x + 512
        start_y = np.random.randint(low=0, high=(self.img_h - 512))
        end_y = start_y + 512

        gt_sisr = gt_sisr[start_y:end_y, start_x:end_x] # (shape: (512, 512, 4))
        gt_sssr = gt_sssr[start_y:end_y, start_x:end_x] # (shape: (512, 512))
        
        gt_sisr_n = gt_sisr # (shape: (512, 512, 4))
        ################################################################################################################
        # resize gt_sisr without interpolation to obtain img (the one used for training in both SSSR and SISR branches)
        ################################################################################################################
        image = resize(gt_sisr_n, (self.new_img_h, self.new_img_w), preserve_range=True, anti_aliasing=True) # (shape: (256, 256, 4))
            
        ################################################################################################################
        # normalize the img (with the mean and std):
        ################################################################################################################
        image = image / ((2**16)-1)
        mean = np.mean(image, axis=(0,1))
        std = np.std(image, axis=(0,1))
        image = image - mean
        image = image / std
        image = image.astype(np.float32)
        image = np.transpose(image, (2, 0, 1)) # (shape: (4, 256, 256))

        gt_sisr_n = gt_sisr_n / ((2**16)-1)
        gt_sisr_n = gt_sisr_n - mean
        gt_sisr_n = gt_sisr_n / std
        gt_sisr_n = gt_sisr_n.astype(np.float32)
        gt_sisr_n = np.transpose(gt_sisr_n, (2, 0, 1)) # (shape: (4, 512, 512)) 
        gt_sisr = np.transpose(gt_sisr, (2, 0, 1)) # (shape: (4, 1000, 1000))
        ################################################################################################################
        # convert numpy -> torch:
        ################################################################################################################
        image = torch.from_numpy(image) # (shape: (4, 256, 256))
        gt_sisr_n = torch.from_numpy(gt_sisr_n) # (shape: (4, 512, 512))
        gt_sssr = torch.from_numpy(gt_sssr) # (shape: (512, 512))
        
        gt_sisr = gt_sisr.astype(np.int16)
        
        data = {
            "img" : image,
            "mean" : mean,
            "std" : std,
            "gt_sisr_n" : gt_sisr_n,
            "gt_sisr" : gt_sisr,
            "gt_sssr" : gt_sssr
        }
        return data

    def __len__(self):
        return self.num_examples