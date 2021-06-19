import dataset
import metrics
import utils

from deeplabv3.model_v4_PixelShuffle import DeepLabV3Plus
import encoders
from typing import Optional

import torch
from torch.optim import lr_scheduler
from torch import nn, optim
from torch.autograd import Variable
import time
import copy
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
from torchmetrics import IoU

############################################################################################################################################
############# Load the Dataset #############################################################################################################
############################################################################################################################################
train_images_path = "/home/usuaris/imatge/sauc.abadal/Corine/data/train_images" 
train_labels_path = "/home/usuaris/imatge/sauc.abadal/Corine/data/train_labels2/train_labels" 

val_images_path = "/home/usuaris/imatge/sauc.abadal/Corine/data/val_images"
val_labels_path = "/home/usuaris/imatge/sauc.abadal/Corine/data/val_labels2/val_labels" 

train_dataset = dataset.DatasetCorine(train_images_path, train_labels_path)
val_dataset = dataset.DatasetCorine(val_images_path, val_labels_path)

train_batch_size = 8
valid_batch_size = 8

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4, drop_last=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=valid_batch_size, shuffle=False, num_workers=4)

num_train_batches = math.ceil(len(train_dataset)/train_batch_size)
num_val_batches = math.ceil(len(val_dataset)/valid_batch_size)

with open("/mnt/gpid07/imatge/sauc.abadal/Corine/data/class_weights.pkl", "rb") as file: # (needed for python3)
    class_weights = np.array(pickle.load(file))
class_weights = class_weights[:-1]
class_weights = torch.from_numpy(class_weights)
class_weights = Variable(class_weights.type(torch.FloatTensor)).cuda()

############################################################################################################################################
############# METRICS ######################################################################################################################
############################################################################################################################################
mIoU = metrics.Jaccard()
psnr = metrics.PSNR()
rmse = metrics.RMSE()
ssim = metrics.SSIM()

############################################################################################################################################
########### LOSS AND OPTIMIZER #############################################################################################################
############################################################################################################################################
l_ce = nn.CrossEntropyLoss(weight=class_weights)
l_mse = nn.MSELoss()

# w1 and w2 set to make the loss value ranges comparable
w1 = 1.0
w2 = 1.0

def loss_func(sssr, sisr, gt_sssr, gt_sisr, sssr_FA, train=True):
    sssr_sim_mt = utils.similarity_matrix(sssr_FA, train)
    sisr_sim_mt = utils.similarity_matrix(sisr.detach(), train)
    loss = l_ce(sssr, gt_sssr) + w1*l_mse(sisr, gt_sisr) + w2*utils.FA_Loss(sssr_sim_mt, sisr_sim_mt)
    return loss

model = DeepLabV3Plus(
            encoder_name = "resnet101",
            encoder_depth = 5,
            encoder_weights = "imagenet",
            encoder_output_stride = 16,
            decoder_channels = 128,
            decoder_atrous_rates = (12, 24, 36),
            in_channels = 4,
            classes = 14,
            upsampling = 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
iou = IoU(num_classes=14).to(device)

lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr)

scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

PATH = "./Corine/model/full_dataset/checkpoints/Corine_v4_PixelShuffle_MSE_1_0_lr_1e-4_NO_Scheduler_0.pth"
model, optimizer, scheduler, start_epoch, train_loss, valid_loss, train_w_iou, train_m_iou, valid_w_iou, valid_m_iou, train_rmse, valid_rmse = utils.load_checkpoint(model, optimizer, scheduler, PATH)

############################################################################################################################################
############### TRAINING LOOP ##############################################################################################################
############################################################################################################################################
def train_model(model, loss_func, optimizer, scheduler, start_epoch, train_loss, valid_loss, train_w_iou, train_m_iou, valid_w_iou, valid_m_iou, train_rmse, valid_rmse, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    if valid_m_iou:
        best_iou = valid_m_iou[-1]
    else:
        best_iou = 0.0
    print(f'Best valid_iou until now: {best_iou:.4f}')
    end_epoch = start_epoch + num_epochs
    for epoch in range(start_epoch, end_epoch):
        print()
        print(f'[Epoch {epoch+1}/{end_epoch}]')
        print('-' * 10)
        model.train() # set model to training mode

        train_loss_b = []
        train_w_iou_b = []
        train_m_iou_b = []
        train_rmse_b = []
        for batch_idx, batch in enumerate(train_loader, 0):     

            optimizer.zero_grad()

            img = batch["img"].to(device)
            gt_sisr = batch["gt_sisr_n"].to(device)
            gt_sssr = batch["gt_sssr"].to(device).long()

            sssr, sisr, sssr_FA = model(img)

            # Segmentation metrics
            sssr_preds = torch.argmax(sssr, dim=1)
            w_iou, _ = mIoU(sssr, gt_sssr)
            train_w_iou_b.append(w_iou.item() * img.shape[0]) # append batchwise IoU weighted by the bacth size
            
            tr_m_iou = iou(sssr_preds, gt_sssr)

            # Super-resolution metrics
            root_mse = rmse(sisr, gt_sisr)
            train_rmse_b.append(root_mse.item() * img.shape[0])

            loss = loss_func(sssr, sisr, gt_sssr, gt_sisr, sssr_FA, train=True)
            train_loss_b.append(loss.item() * img.shape[0]) # append batchwise IoU weighted by the bacth size

            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                 print(f'Batch {batch_idx+1} / {num_train_batches}, Training wIoU = {w_iou.item():.4f}, mIoU = {tr_m_iou.item():.4f},Loss = {loss.item():.4f}, RMSE = {root_mse.item():.4f}')
        
        tr_m_iou = iou.compute()
        
        train_loss.append(np.sum(train_loss_b)/train_loader.dataset.num_examples)
        train_w_iou.append(np.sum(train_w_iou_b)/train_loader.dataset.num_examples)
        train_m_iou.append(tr_m_iou.item())
        iou.reset()
        
        train_rmse.append(np.sum(train_rmse_b)/train_loader.dataset.num_examples)
        print(f'-END of Epoch- Training wIoU = {train_w_iou[-1]:.4f}, mIoU = {train_m_iou[-1]:.4f}, Loss = {train_loss[-1]:.4f}, RMSE = {train_rmse[-1]:.4f}')
        scheduler.step()

        model.eval() # set model to validation mode
        valid_loss_b = []
        valid_w_iou_b = []
        valid_m_iou_b = []
        valid_rmse_b = []
        for batch_idx, batch in enumerate(val_loader, 0): 
            with torch.no_grad():    
                
                img = batch["img"].to(device)
                gt_sisr = batch["gt_sisr_n"].to(device)
                gt_sssr = batch["gt_sssr"].to(device).long()

                sssr, sisr, sssr_FA = model(img)

                # Segmentation metrics
                sssr_preds = torch.argmax(sssr, dim=1)
                w_iou, _ = mIoU(sssr, gt_sssr)
                valid_w_iou_b.append(w_iou.item() * img.shape[0]) # append batchwise IoU weighted by the bacth size
                
                val_m_iou = iou(sssr_preds, gt_sssr)

                # Super-resolution metrics
                root_mse = rmse(sisr, gt_sisr)
                valid_rmse_b.append(root_mse.item() * img.shape[0])

                loss = loss_func(sssr, sisr, gt_sssr, gt_sisr, sssr_FA, train=True)
                valid_loss_b.append(loss.item() * img.shape[0]) # append batchwise IoU weighted by the bacth size
                
                if batch_idx % 100 == 0:
                     print(f'Batch {batch_idx+1} / {num_val_batches}')
        print()
        val_m_iou = iou.compute()
        
        valid_loss.append(np.sum(valid_loss_b)/val_loader.dataset.num_examples)
        valid_w_iou.append(np.sum(valid_w_iou_b)/val_loader.dataset.num_examples)
        valid_m_iou.append(val_m_iou.item())
        iou.reset()
        
        valid_rmse.append(np.sum(valid_rmse_b)/val_loader.dataset.num_examples)
        print(f'Validation wIoU = {valid_w_iou[-1]:.4f}, mIoU = {valid_m_iou[-1]:.4f}, Loss = {valid_loss[-1]:.4f}. RMSE = {valid_rmse[-1]:.4f}')

        # deep copy the model 
        if valid_m_iou[-1] > best_iou:
            best_iou = valid_m_iou[-1] 
            best_model_wts = copy.deepcopy(model.state_dict())
            state = {
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss':  train_loss,
                'train_w_iou': train_w_iou,
                'train_m_iou': train_m_iou,
                'train_rmse': train_rmse,
                'valid_loss': valid_loss,
                'valid_w_iou': valid_w_iou,
                'valid_m_iou': valid_m_iou,
                'valid_rmse': valid_rmse
            }
            torch.save(state, "./Corine/model/full_dataset/checkpoints/Corine_v4_PixelShuffle_MSE_1_0_lr_1e-4_NO_Scheduler_1.pth")
        print() # end of epoch

    print('Finished Training!')
    time_elapsed = time.time() - since
    print(f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')
    model.load_state_dict(best_model_wts)
    
    fig1, ax1 = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    ax1.plot(range(0, end_epoch), train_loss, label="Training Loss")
    ax1.plot(range(0, end_epoch), valid_loss, label="Validation Loss")
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training vs Validation Loss')
    ax1.legend()
    plt.close(fig1)
    fig1.savefig('./Corine/model/full_dataset/Plots/loss_Corine_v4_PixelShuffle_MSE_1_0_lr_1e-4_NO_Scheduler_1.png')

    fig2, ax2 = plt.subplots( nrows=1, ncols=1 )  # create figure & 2 axis
    ax2.plot(range(0, end_epoch), train_w_iou, label="Training wIoU")
    ax2.plot(range(0, end_epoch), valid_w_iou, label="Validation wIoU")
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('wIoU')
    ax2.set_title('Training vs Validation wIoU')
    ax2.legend()
    plt.close(fig2)
    fig2.savefig('./Corine/model/full_dataset/Plots/w_iou_Corine_v4_PixelShuffle_MSE_1_0_lr_1e-4_NO_Scheduler_1.png')

    fig3, ax3 = plt.subplots( nrows=1, ncols=1 )  # create figure & 3 axis
    ax3.plot(range(0, end_epoch), train_m_iou, label="Training mIoU")
    ax3.plot(range(0, end_epoch), valid_m_iou, label="Validation mIoU")
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('mIoU')
    ax3.set_title('Training vs Validation mIoU')
    ax3.legend()
    plt.close(fig3)
    fig3.savefig('./Corine/model/full_dataset/Plots/m_iou_Corine_v4_PixelShuffle_MSE_1_0_lr_1e-4_NO_Scheduler_1.png')

    fig4, ax4 = plt.subplots( nrows=1, ncols=1 )  # create figure & 4 axis
    ax4.plot(range(0, end_epoch), train_rmse, label="Training RMSE")
    ax4.plot(range(0, end_epoch), valid_rmse, label="Validation RMSE")
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('RMSE')
    ax4.set_title('Training vs Validation RMSE')
    ax4.legend()
    plt.close(fig4)
    fig4.savefig('./Corine/model/full_dataset/Plots/rmse_Corine_v4_PixelShuffle_MSE_1_0_lr_1e-4_NO_Scheduler_1.png')
    
    return model

model = train_model(model, loss_func, optimizer, scheduler, start_epoch, train_loss, valid_loss, train_w_iou, train_m_iou, valid_w_iou, valid_m_iou, train_rmse, valid_rmse, num_epochs=90)
torch.save(model.state_dict(), "./Corine/model/full_dataset/trained_models/Corine_v4_PixelShuffle_MSE_1_0_lr_1e-4_NO_Scheduler_1.pth")