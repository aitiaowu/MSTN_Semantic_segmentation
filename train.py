import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import cv2
import dataset
from datasetvessel import vesselDataset
from modelvessel import AlexNet
import utils
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument('--s', default=0, type=int)
parser.add_argument('--t', default=1, type=int)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
cuda = torch.cuda.is_available()
#writer = SummaryWriter()


def adaptation_factor(x):
	if x>= 1.0:
		return 1.0
	den = 1.0 + math.exp(-10 * x)
	lamb = 2.0 / den - 1.0
	return lamb

def lr_schedule(opt, epoch, mult):
    lr = init_lr / pow(1 + 0.001 * epoch, 0.75)
    for ind, param_group in enumerate(opt.param_groups):
        param_group['lr'] = lr * mult[ind]
    return lr

if __name__ == '__main__':

    init_lr = 0.01
    batch_size = 2
    max_epoch = 10000000
    lr_mult = [0.1, 0.2, 1, 2]
    lr_mult_D = [1, 2]

    CS_weights = np.array( (0.005, 0.41, 0.46, 1), dtype=np.float32 )
    CS_weights = torch.from_numpy(CS_weights)

    CS_weights_s = np.array( (0.02, 0.82, 0.6923, 1.1166), dtype=np.float32 )
    CS_weights_s = torch.from_numpy(CS_weights_s)
    s_list_path = '/content/sourcelist.txt'
    t_list_path = '/content/targetlist.txt'
    s_folder_path = '/content/DATA/older_Data'
    t_folder_path = '/content/DATA/young_Data'
    n_class = 4

    #pretrain_path = 'checkpoint/' + resume + '.pth'
    checkpoint_save_path = '/content/drive/MyDrive/Study/Domain_Adaptation_Vessel/MSTN/mstn-master/checkpoint'

    #img = cv2.imread('./dataset/office/amazon/images/headphones/frame_0037.jpg')
    #print(s_list_path ,t_list_path)
    #print(img)
    #aaaaaa
    s_dataset = vesselDataset(s_folder_path, s_list_path)
    t_dataset = vesselDataset(t_folder_path, t_list_path)
    #print(len(s_loader),len(t_loader))
    #aaa

    s_loader = torch.utils.data.DataLoader(s_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    t_loader = torch.utils.data.DataLoader(t_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    s_loader_iterations = 0
    t_loader_iterations = 0

    s_loader_epoch = iter(s_loader)
    t_loader_epoch = iter(t_loader)

    s_loader_len, t_loader_len = len(s_loader), len(t_loader)
    #print(s_loader_len, t_loader_len)
    

    model=AlexNet(n_class=n_class)
    model.cuda()


    opt, opt_D = model.get_optimizer(init_lr, lr_mult, lr_mult_D)
    class_weights = Variable(CS_weights).cuda()
    class_weights_s = Variable(CS_weights_s).cuda()

    epoch = 0


    for epoch in range(epoch, 150000):
        epoch += 1
        model.train()
        lamb = adaptation_factor(epoch * 1.0 / max_epoch)
        current_lr, _ = lr_schedule(opt, epoch, lr_mult), lr_schedule(opt_D, epoch, lr_mult_D)
        
        if epoch % s_loader_len == 0:
            s_loader_epoch = iter(s_loader)
        if epoch % t_loader_len == 0:
            t_loader_epoch = iter(t_loader)

        xs, ys ,_ = next(s_loader_epoch)
        xt, yt ,_ = next(t_loader_epoch)
        
        #domain lbl
        '''
        # get xs, ys, xt
      # If the s_loader iterator needs to be reset
        if s_loader_iterations >= len(s_loader):
            s_loader_epoch = iter(s_loader)
            s_loader_iterations = 0

        xs, ys = next(s_loader_epoch)
        s_loader_iterations += 1

        # If the t_loader iterator needs to be reset
        if t_loader_iterations >= len(t_loader):
            t_loader_epoch = iter(t_loader)
            t_loader_iterations = 0

        xt, yt = next(t_loader_epoch)
        t_loader_iterations += 1
        '''

        B, C, H, W = xs.shape
        #src_domain_lbl = torch.zeros(B)
        #trg_domain_lbl = torch.ones(B)
        
        xs, ys,  = Variable(xs).cuda(), Variable(ys.long()).cuda()
        xt, yt  = Variable(xt).cuda(), Variable(yt.long()).cuda()

        # forward
        s_feature, s_score, s_pred = model.forward(xs) # feature -> D(x); score -> classification score(31)
        t_feature, t_score, t_pred = model.forward(xt) # pred -> classifier(fake label)
        

        
        #convert to cuda
        

        C_loss_src = model.c_loss(s_pred, ys, class_weights) #### improve the classifier's performance on segmentation
        C_loss_trg = model.c_loss(t_pred, yt, class_weights) 

        #calculate loss
        # discriminator (1)
        s_logit, t_logit = model.forward_D(s_feature), model.forward_D(t_feature)
        #s_logit, t_logit = model.forward_D(s_pred), model.forward_D(t_pred)
        
        #according to centoid to get semantic loss. to decrease the gap between two domains
        G_loss, D_loss = model.adloss(s_logit,t_logit)

        #semantic_loss = model.smloss(s_feature, t_feature, ys, yt)
        semantic_loss = model.smloss(s_feature, t_feature, ys, t_pred)
        Dregloss, Gregloss = model.regloss()
        #improve the classifier's performance
        F_loss = C_loss_src + Gregloss + lamb * G_loss + lamb * semantic_loss
        #F_loss = C_loss + Gregloss #+ lamb * G_loss + lamb * semantic_loss
        D_loss = D_loss + Dregloss + 0.1 * C_loss_trg

        #error test


        # Zero the gradients computed for each weight
        opt_D.zero_grad()
        #Backward pass through the network
        D_loss.backward(retain_graph=True)
        # Update the weights
        opt_D.step()
        opt.zero_grad()
        #error line

        F_loss.backward(retain_graph=True)
        opt.step()
        '''
        writer.add_scalar("Loss_src_val/train",C_loss_src.data, epoch)
        writer.add_scalar("Loss_trg_val/train", C_loss_trg.data, epoch)
        writer.add_scalar("Loss/all", F_loss.data, epoch)
        writer.flush()
        '''
        if epoch % 10 == 0:  ## Show loss every "print_freq" times

                #print('[it %d][src_seg_loss %.4f][loss_all %.4f][loss_val %.4f]' %\(i+1, loss_seg_src.data, loss_all.data, loss_val.data))
          print('[it %d][src seg loss %.4f][domain loss %.4f][loss_val %.4f][classifier seg loss %.4f]  '%  
          (epoch, C_loss_src.data, D_loss.data ,C_loss_trg.data, F_loss.data))

            # save model
        if epoch >= 100000 and C_loss_src.data < 0.01:
        #if epoch >= 1:
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'opt': opt.state_dict(),
                'opt_D': opt_D.state_dict()
            }, os.path.join(checkpoint_save_path, 'mstn_module' + str(epoch) + '.pth'))
            print('saving checkpoint')
'''
        # validation
        if epoch % 100 == 0 and epoch != 0:
            print('    =======    START VALIDATION    =======    ')
            model.eval()
            v_correct, v_sum = 0, 0
            zeros, zeros_classes = torch.zeros(n_class), torch.zeros(n_class)

            for ind2, (xv, yv) in enumerate(val_loader):

                v_feature, v_score, v_pred = model.forward(xv)
                v_pred_label = torch.max(v_score, 1)[1]
                v_equal = torch.eq(v_pred_label, yv).float()
                zeros = zeros.scatter_add(0, yv, v_equal)
                zeros_classes = zeros_classes.scatter_add(0, yv, torch.ones_like(yv, dtype=torch.float))
                v_correct += torch.sum(v_equal).item()
                v_sum += len(yv)
            v_acc = v_correct / v_sum
            print('validation: {}, {}'.format(v_correct, v_acc, zeros))
            print('class: {}'.format(zeros.tolist()))
            print('class: {}'.format(zeros_classes.tolist()))
            print('source: {}, target: {}, batch_size: {}, init_lr: {}'.format(s_name, t_name, batch_size, init_lr))
            print('lr_mult: {}, lr_mult_D: {}'.format(lr_mult, lr_mult_D))
            print('    =======    START TRAINING    =======    ')



'''

