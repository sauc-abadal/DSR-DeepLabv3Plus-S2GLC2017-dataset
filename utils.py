import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torchvision
import os
import earthpy.plot as ep
import matplotlib
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
from skimage.transform import resize

def label2class():
    
    classes = np.array(["Clouds","Artificial surfaces and constructions","Cultivated areas",
                       "Vineyards","Broadleaf tree cover",
                      "Coniferous tree cover","Herbaceous vegetation",
                      "Moors and Heathland","Sclerophyllous vegetation",
                     "Marshes","Peatbogs","Natural material surfaces",
                      "Permanent snow covered surfaces",
                      "Water bodies",
                       "No data"
                      ])
    return classes

def unnormalize(img, data): # function to un-normalize an image, expected tensor with shape (Ch, H, W)
    
    mean = data["mean"]
    std = data["std"]
    
    image = img.numpy()
    mean_expanded = np.expand_dims(np.expand_dims(std, -1), -1)
    std_expanded = np.expand_dims(np.expand_dims(std, -1), -1)
    img_un = (image * std_expanded + mean_expanded) * ((2**16)-1)
    img_un = np.clip(img_un, 0, ((2**16)-1)).astype(np.uint16)
    
    return img_un

def imshow_corine(data): # function to visualize both the ground truth for Semantic Segmentation and Super Resolution
        
    gt_sssr = data["gt_sssr"]
    gt_sisr_n = data["gt_sisr_n"]
    
    gt_sisr_un = unnormalize(gt_sisr_n, data)
    fig = plt.figure(figsize=(20,20))
    ax = plt.subplot(1,2,1)
    ep.plot_rgb(gt_sisr_un, rgb=(2,1,0), stretch=True, ax=ax)
    
    # https://stackoverflow.com/questions/14777066/matplotlib-discrete-colorbar
    # # Let's design a dummy land use field
    # A = np.reshape([7,2,13,7,2,2], (2,3))
    # vals = np.unique(A)
    # Let's also design our color mapping: 1s should be plotted in blue, 2s in red, etc...
    col_dict={0:[1.0,1.0,1.0,1.0],
              1:[210.0/255.0,0.0/255.0,0.0/255.0,255.0/255.0],
              2:[253/255.0,211/255.0,39/255.0,255/255.0],
              3:[176/255.0,91/255.0,16/255.0,255/255.0],
              4: [35/255.0,152/255.0,0/255.0,255/255.0],
              5:[8/255.0,98/255.0,0/255.0,255/255.0],
              6:[249/255.0,150/255.0,39/255.0,255/255.0],
              7:[141/255.0,139/255.0,0/255.0,255/255.0],
              8:[95/255.0,53/255.0,6/255.0,255/255.0],
              9:[149/255.0,107/255.0,196/255.0,255/255.0],
              10:[77/255.0,37/255.0,106/255.0,255/255.0],
              11:[154/255.0,154/255.0,154/255.0,255/255.0],
              12:[106/255.0,255/255.0,255/255.0,255/255.0],
              13:[20/255.0,69/255.0,249/255.0,255/255.0],
              14:[20/255.0,69/255.0,249/255.0,10/255.0]
             }
    
    # We create a colormar from our list of colors
    cm = ListedColormap([col_dict[x] for x in col_dict.keys()])

    # Let's also define the description of each category : 1 (blue) is Sea; 2 (red) is burnt, etc... Order should be respected here ! Or using another dict maybe could help.
    labels = np.array(["Clouds","Artificial surfaces and constructions","Cultivated areas",
                       "Vineyards","Broadleaf tree cover",
                      "Coniferous tree cover","Herbaceous vegetation",
                      "Moors and Heathland","Sclerophyllous vegetation",
                     "Marshes","Peatbogs","Natural material surfaces",
                      "Permanent snow covered surfaces",
                      "Water bodies",
                       "No data"
                      ])
    len_lab = len(labels)

    # prepare normalizer
    ## Prepare bins for the normalizer
    norm_bins = np.sort([*col_dict.keys()]) + 0.5
    norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
    #print(norm_bins)
    ## Make normalizer and formatter
    norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)
    #fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

    # Plot our figure
    #fig,ax = plt.subplots(figsize=(10,10))
    ax = plt.subplot(1,2,2)
    im = ax.imshow(gt_sssr, cmap=cm, norm=norm)
    ax.axis('off')

    #diff = norm_bins[1:] - norm_bins[:-1]
    #tickz = norm_bins[:-1] + diff / 2
    #cb = fig.colorbar(im, format=fmt, ticks=tickz)
    plt.show()
    
def visualize_results_SISR(data, sisr):  
    
    img_n = data["img"]
    img_un = unnormalize(img_n[0].cpu().detach(), data)
    
    gt_sisr = data["gt_sisr"][0].cpu().numpy() # un-normalized SISR ground truth
    
    sisr_un = unnormalize(sisr[0].cpu().detach(), data)
    
    ############################################################################
    ###################### SISR BRANCH RESULTS #################################
    ############################################################################
    patch_size = 70
    start_x = np.random.randint(low=0, high=(512 - patch_size))
    #start_x = 135
    end_x = start_x + patch_size
    start_y = np.random.randint(low=0, high=(512 - patch_size))
    #start_y = 187
    end_y = start_y + patch_size
    
    # SISR Ground Truth
    fig = plt.figure(figsize=(20,20))
    ax = plt.subplot(1,3,1)
    ep.plot_rgb(gt_sisr, rgb=(2,1,0), stretch=True, ax=ax)
    ax.title.set_text('SISR Ground Truth')
    # Create a Rectangle patch
    rect = patches.Rectangle((start_x, start_y), patch_size, patch_size, linewidth=1, edgecolor='r', facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)
    
    # Training image
    ax = plt.subplot(1,3,2)
    img_un_NN = resize(img_un[0].transpose(1,2,0), (512,512), order=0, preserve_range=True, anti_aliasing=True)
    ep.plot_rgb(img_un_NN.transpose(2,0,1), rgb=(2,1,0), stretch=True, ax=ax)
    ax.title.set_text('Training Image')
    # Create a Rectangle patch
    rect = patches.Rectangle((start_x, start_y), patch_size, patch_size, linewidth=1, edgecolor='r', facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)
    
    # Reconstructed image
    ax = plt.subplot(1,3,3)
    ep.plot_rgb(sisr_un[0], rgb=(2,1,0), stretch=True, ax=ax)
    ax.title.set_text('Reconstructed Image')
    # Create a Rectangle patch
    rect = patches.Rectangle((start_x, start_y), patch_size, patch_size, linewidth=1, edgecolor='r', facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)
    
    #########################
    fig = plt.figure(figsize=(20,20))
    ax2 = plt.subplot(1,4,1)
    patch_gt_sisr = gt_sisr[:, start_y:end_y, start_x:end_x]
    ep.plot_rgb(patch_gt_sisr, rgb=(2,1,0), stretch=True, ax=ax2)
    ax2.title.set_text(f'{patch_size}x{patch_size} patch from \n SISR Ground Truth x={start_x}, y={start_y}')
    
    ax2 = plt.subplot(1,4,2)
    patch_img_un = img_un[0, :, start_y//2:end_y//2, start_x//2:end_x//2]
    patch_nearest = resize(patch_img_un.transpose(1,2,0), (patch_size,patch_size), order=0, preserve_range=True, anti_aliasing=True)
    ep.plot_rgb(patch_nearest.transpose(2,0,1), rgb=(2,1,0), stretch=True, ax=ax2)
    ax2.title.set_text(f'{patch_size}x{patch_size} patch from \n Training Image NN x={start_x}, y={start_y}')
    
    ax2 = plt.subplot(1,4,3)
    patch_bicubic = resize(patch_img_un.transpose(1,2,0), (patch_size,patch_size), order=3, preserve_range=True, anti_aliasing=True)
    ep.plot_rgb(patch_bicubic.transpose(2,0,1), rgb=(2,1,0), stretch=True, ax=ax2)
    ax2.title.set_text(f'{patch_size}x{patch_size} patch from \n Training Image Bicubic x={start_x}, y={start_y}')
    
    ax2 = plt.subplot(1,4,4)
    patch_sisr_un = sisr_un[0, :, start_y:end_y, start_x:end_x]
    ep.plot_rgb(patch_sisr_un, rgb=(2,1,0), stretch=True, ax=ax2)
    ax2.title.set_text(f'{patch_size}x{patch_size} patch from \n Reconstructed Image x={start_x}, y={start_y}')
    
    #ax.axis('off')
    plt.show()

def visualize_results_SSSR(data, sssr):   
    patch_size = 70
    start_x = np.random.randint(low=0, high=(512 - patch_size))
    #start_x = 381
    end_x = start_x + patch_size
    start_y = np.random.randint(low=0, high=(512 - patch_size))
    #start_y = 421
    end_y = start_y + patch_size
    
    gt_sisr = data["gt_sisr"][0].cpu().numpy() # un-normalized SISR ground truth
    fig = plt.figure(figsize=(20,20))
    ax = plt.subplot(1,3,1)
    ep.plot_rgb(gt_sisr, rgb=(2,1,0), stretch=True, ax=ax)
    ax.title.set_text('SISR Ground Truth')
    # Create a Rectangle patch
    #rect = patches.Rectangle((start_x, start_y), patch_size, patch_size, linewidth=1, edgecolor='r', facecolor='none')
    # Add the patch to the Axes
    #ax.add_patch(rect)
    
    gt_sssr = data["gt_sssr"][0].cpu().numpy()
    
    sssr_preds = torch.argmax(sssr, dim=1)
    sssr_preds = sssr_preds[0].cpu().detach().numpy()
    
    ############################################################################
    ###################### SSSR BRANCH RESULTS #################################
    ############################################################################
    # SSSR Ground Truth
   # https://stackoverflow.com/questions/14777066/matplotlib-discrete-colorbar
    # # Let's design a dummy land use field
    # A = np.reshape([7,2,13,7,2,2], (2,3))
    # vals = np.unique(A)
    # Let's also design our color mapping: 1s should be plotted in blue, 2s in red, etc...
    col_dict={0:[1.0,1.0,1.0,1.0],
              1:[210.0/255.0,0.0/255.0,0.0/255.0,255.0/255.0],
              2:[253/255.0,211/255.0,39/255.0,255/255.0],
              3:[176/255.0,91/255.0,16/255.0,255/255.0],
              4: [35/255.0,152/255.0,0/255.0,255/255.0],
              5:[8/255.0,98/255.0,0/255.0,255/255.0],
              6:[249/255.0,150/255.0,39/255.0,255/255.0],
              7:[141/255.0,139/255.0,0/255.0,255/255.0],
              8:[95/255.0,53/255.0,6/255.0,255/255.0],
              9:[149/255.0,107/255.0,196/255.0,255/255.0],
              10:[77/255.0,37/255.0,106/255.0,255/255.0],
              11:[154/255.0,154/255.0,154/255.0,255/255.0],
              12:[106/255.0,255/255.0,255/255.0,255/255.0],
              13:[20/255.0,69/255.0,249/255.0,255/255.0],
              14:[20/255.0,69/255.0,249/255.0,10/255.0]
             }
    
    # We create a colormar from our list of colors
    cm = ListedColormap([col_dict[x] for x in col_dict.keys()])

    # Let's also define the description of each category : 1 (blue) is Sea; 2 (red) is burnt, etc... Order should be respected here ! Or using another dict maybe could help.
    labels = np.array(["Clouds","Artificial surfaces and constructions","Cultivated areas",
                       "Vineyards","Broadleaf tree cover",
                      "Coniferous tree cover","Herbaceous vegetation",
                      "Moors and Heathland","Sclerophyllous vegetation",
                     "Marshes","Peatbogs","Natural material surfaces",
                      "Permanent snow covered surfaces",
                      "Water bodies",
                       "No data"
                      ])
    len_lab = len(labels)

    # prepare normalizer
    ## Prepare bins for the normalizer
    norm_bins = np.sort([*col_dict.keys()]) + 0.5
    norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
    #print(norm_bins)
    ## Make normalizer and formatter
    norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)
    #fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

    # Plot our figure
    ax = plt.subplot(1,3,2)
    im = ax.imshow(gt_sssr, cmap=cm, norm=norm)
    ax.title.set_text('SSSR Ground Truth')
    # Create a Rectangle patch
    #rect = patches.Rectangle((start_x, start_y), patch_size, patch_size, linewidth=1, edgecolor='r', facecolor='none')
    # Add the patch to the Axes
    #ax.add_patch(rect)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    ax = plt.subplot(1,3,3)
    im = ax.imshow(sssr_preds, cmap=cm, norm=norm)
    ax.title.set_text('Predicted Mask')
    # Create a Rectangle patch
    #rect = patches.Rectangle((start_x, start_y), patch_size, patch_size, linewidth=1, edgecolor='r', facecolor='none')
    # Add the patch to the Axes
    #ax.add_patch(rect)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    #########################
    fig = plt.figure(figsize=(20,20))
    ax2 = plt.subplot(1,3,1)
    patch_gt_sisr = gt_sisr[:, start_y:end_y, start_x:end_x]
    ep.plot_rgb(patch_gt_sisr, rgb=(2,1,0), stretch=True, ax=ax2)
    ax2.title.set_text(f'{patch_size}x{patch_size} patch from \n SISR Ground Truth x={start_x}, y={start_y}')
    
    ax2 = plt.subplot(1,3,2)
    patch_gt_sssr = gt_sssr[start_y:end_y, start_x:end_x]
    im = ax2.imshow(patch_gt_sssr, cmap=cm, norm=norm)
    ax2.title.set_text(f'{patch_size}x{patch_size} patch from \n SSSR Ground Truth x={start_x}, y={start_y}')
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    
    ax2 = plt.subplot(1,3,3)
    patch_sssr_preds = sssr_preds[start_y:end_y, start_x:end_x]
    im = ax2.imshow(patch_sssr_preds, cmap=cm, norm=norm)
    ax2.title.set_text(f'{patch_size}x{patch_size} patch from \n Predicted Mask x={start_x}, y={start_y}')
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)

    #diff = norm_bins[1:] - norm_bins[:-1]
    #tickz = norm_bins[:-1] + diff / 2
    #cb = fig.colorbar(im, format=fmt, ticks=tickz)
    plt.show()
    
    
def similarity_matrix(f, train=True):
    # f expected shape (Bs, C', H', W')
    if train:
        f = F.interpolate(f, size=(f.shape[2] // 8, f.shape[3] // 8), mode='nearest')   # before computing the relationship of every pair of pixels, 
                                                                                        # subsample the feature map to its 1/8
    else:
         f = F.interpolate(f, size=(f.shape[2] // 16, f.shape[3] // 16), mode='nearest')   # before computing the relationship of every pair of pixels, 
                                                                                        # subsample the feature map to its 1/16
    f = f.permute((0,2,3,1))
    f = torch.reshape(f, (f.shape[0], -1, f.shape[3])) # shape (Bs, H'xW', C')
    f_n = torch.linalg.norm(f, ord=None, dim=2).unsqueeze(-1) # ord=None indicates 2-Norm, 
                                                #unsqueeze last dimension to broadcast later
    eps = 1e-8
    f_norm = f / torch.max(f_n, eps * torch.ones_like(f_n))
    sim_mt = f_norm @ f_norm.transpose(2, 1)
    return sim_mt

def FA_Loss(sssr_sim_mt, sisr_sim_mt):
    nelem = sssr_sim_mt.shape[1] * sssr_sim_mt.shape[2]
    dist = torch.abs(sssr_sim_mt - sisr_sim_mt)
    l_fa = 1/nelem * torch.sum(dist, dim=[1, 2])
    return l_fa.mean()

def load_checkpoint(model, optimizer, scheduler, PATH):
    if os.path.isfile(PATH):
        print("=> loading checkpoint '{}'".format(PATH))
        checkpoint = torch.load(PATH)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        train_loss = checkpoint['train_loss']
        valid_loss = checkpoint['valid_loss']
        train_w_iou = checkpoint['train_w_iou']
        train_m_iou = checkpoint['train_m_iou']
        valid_w_iou = checkpoint['valid_w_iou']
        valid_m_iou = checkpoint['valid_m_iou']
        train_rmse = checkpoint['train_rmse']
        valid_rmse = checkpoint['valid_rmse']
        print("=> loaded checkpoint '{}' (epoch {})".format(PATH, checkpoint['epoch']))     
    else:
        print("=> no checkpoint found at '{}'".format(PATH))
        start_epoch = 0
        train_loss = []
        valid_loss = []
        train_w_iou = []
        train_m_iou = []
        valid_w_iou = []
        valid_m_iou = []
        train_rmse = []
        valid_rmse = []
    
    return model, optimizer, scheduler, start_epoch, train_loss, valid_loss, train_w_iou, train_m_iou, valid_w_iou, valid_m_iou, train_rmse, valid_rmse