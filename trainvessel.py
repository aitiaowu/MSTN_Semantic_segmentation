import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import cv2

import dataset
from model import AlexNet
import utils


parser = argparse.ArgumentParser()
parser.add_argument('--s', default=0, type=int)
parser.add_argument('--t', default=1, type=int)
args = parser.parse_args()
#os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
#cuda = torch.cuda.is_available()



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
    batch_size = 16
    max_epoch = 10000000
    lr_mult = [0.1, 0.2, 1, 2]
    lr_mult_D = [1, 2]
    dataset_names = ['amazon', 'webcam', 'dslr']
    s_name = dataset_names[args.s]
    t_name = dataset_names[args.t]
    #print(s_name, t_name)


    s_list_path = './data_list/' + s_name + '_list.txt'
    t_list_path = './data_list/' + t_name + '_list.txt'
    s_folder_path = './dataset/office/' + s_name + '/images'
    t_folder_path = './dataset/office/' + t_name + '/images'
    n_class = 31

    #pretrain_path = 'checkpoint/' + resume + '.pth'
    checkpoint_save_path = './checkpoint/' + '.pth'

    #img = cv2.imread('./dataset/office/amazon/images/headphones/frame_0037.jpg')
    #print(s_list_path ,t_list_path)
    #print(img)
    #aaa
    s_loader = dataset.Office(s_list_path)
    t_loader = dataset.Office(t_list_path)
    #print(len(s_loader),len(t_loader))
    #aaa

    s_loader = torch.utils.data.DataLoader(s_loader,
                                           batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    t_loader = torch.utils.data.DataLoader(t_loader,
                                           batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(dataset.Office(t_list_path, training=False),
                                             batch_size=1, num_workers=8)

    s_loader_len, t_loader_len = len(s_loader), len(t_loader)
    #print(s_loader_len, t_loader_len,len(val_loader))


    model = AlexNet(n_class=n_class)


    opt, opt_D = model.get_optimizer(init_lr, lr_mult, lr_mult_D)

    epoch = 0


    for epoch in range(epoch, 100000):
        epoch += 1
        model.train()
        lamb = adaptation_factor(epoch * 1.0 / max_epoch)
        current_lr, _ = lr_schedule(opt, epoch, lr_mult), lr_schedule(opt_D, epoch, lr_mult_D)
        '''
        if epoch % s_loader_len == 0:
            s_loader_epoch = iter(s_loader)
        if epoch % t_loader_len == 0:
            t_loader_epoch = iter(t_loader)

        xs, ys = s_loader_epoch.next()
        xt, yt = t_loader_epoch.next()
        '''
        s_loader_epoch = iter(s_loader)
        t_loader_epoch = iter(t_loader)

        # get xs, ys, xt
        xs, ys = next(s_loader_epoch)
        xt, yt = next(t_loader_epoch)

        # forward
        s_feature, s_score, s_pred = model.forward(xs) # feature -> D(x); score -> classification score(31)
        t_feature, t_score, t_pred = model.forward(xt) # pred -> classifier(fake label)
        C_loss = model.closs(s_score, ys)

        #calculate loss
        # discriminator (1)
        s_logit, t_logit = model.forward_D(s_feature), model.forward_D(t_feature)

        #according to centoid to get semantic loss. to decrease the gap between two domains
        G_loss, D_loss, semantic_loss = model.adloss(s_logit, t_logit, s_feature, t_feature, ys, t_pred)
        Dregloss, Gregloss = model.regloss()
        #improve the classifier's performance

        F_loss = C_loss + Gregloss + lamb * G_loss + lamb * semantic_loss
        #F_loss = C_loss + Gregloss #+ lamb * G_loss + lamb * semantic_loss
        D_loss = D_loss + Dregloss

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

        if epoch % 10 == 0: #Step {0:<10}
            s_pred_label = torch.max(s_score, 1)[1]
            s_correct = torch.sum(torch.eq(s_pred_label, ys).float())
            #s_acc = torch.div(s_correct, ys.size(0))

            print('epoch: {}, lr: {}, lambda: {}'.format(epoch, current_lr, lamb))

            print('correct: {}, C_loss: {}, G_loss:{}, D_loss:{}, Gregloss: {}, Dregloss: {}, semantic_loss: {}, F_loss: {}'.format(
                s_correct.item(), C_loss.item(), G_loss.item(), D_loss.item(),
                Gregloss.item(), Dregloss.item(), semantic_loss.item(), F_loss.item()))
            #print('the acc is ',  s_acc.item())

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

        # save model
        if epoch % 1000 == 0 and epoch != 0:
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'opt': opt.state_dict(),
                'opt_D': opt_D.state_dict()
            }, checkpoint_save_path)
'''


