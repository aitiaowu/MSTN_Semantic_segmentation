import itertools
from itertools import chain
import torch
import torch.nn as nn
import utils
import torch.nn.functional as F



class DoubleConv(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(DoubleConv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,1,1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)


def skipconnection(x1, x2):
  # input is CHW
  diffY = x2.size()[2] - x1.size()[2]
  diffX = x2.size()[3] - x1.size()[3]

  x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                  diffY // 2, diffY - diffY // 2])
  x = torch.cat([x2, x1], dim=1)
  return x

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class AlexNet(nn.Module):

    def __init__(self, n_class):
        super(AlexNet, self).__init__()
        self.n_channels = 3
        self.n_class = n_class
        self.s_centroid = torch.zeros(self.n_class, 256,5,10).cuda()
        self.t_centroid = torch.zeros(self.n_class, 256,5,10).cuda()
        


        # features
        self.init_encoder = nn.Sequential(
            nn.Conv2d(self.n_channels, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.s1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
        )

        self.s2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
        )
        self.s3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
        )
        self.s4 = nn.Sequential(  ## BottleNeck
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
        )

        self.s5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.down5 = nn.Sequential(
            nn.MaxPool2d(2),
        )
        


        # features_size = (256, 11, 21) c,h,w

        ####### for Fake label #####, Unet-mask upsampling
        #new classifier
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0),
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0),
        )
        self.up6 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0),
        )

        self.e1 = nn.Sequential(
            nn.Conv2d(256 * 2, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.e2 = nn.Sequential(
            nn.Conv2d(128 * 2, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.e3 = nn.Sequential(
            nn.Conv2d(128 * 2, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.e4 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.e5 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.e6 = nn.Sequential(
            nn.Conv2d(32 * 2, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )




        self.outc = nn.Sequential(
          nn.Conv2d(32, self.n_class, kernel_size=1) ## output size[B, 4, h, w]
        )


  #########################################################
        #Descriminator
        self.D = nn.Sequential(
            nn.Linear(3200, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, 2)
        )

        self.MSEloss = nn.MSELoss()
        self.BCEloss = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, x):
        _, _, h, w = x.size()
        s0 = self.init_encoder(x) #s1
        down0 = self.down1(s0) #p1
        s1 = self.s1(down0)  #s2
        down1 = self.down1(s1) #p2
        s2 = self.s2(down1)  #s3
        down2 = self.down2(s2) #p3
        s3 = self.s3(down2)  #s4
        down3 = self.down3(s3) #p4
        s4 = self.s4(down3)
        down4 = self.down4(s4)
        s5 = self.s5(down4)
        down5 = self.down5(s5)

        '''
        down5 = self.down4(s5)
        s6 = self.s6(down5)
        down6 = self.down6(s6)
        features = self.s7(down6)
        '''


        #print('features shape: ', s5.shape)
        #print('s0',s0.shape,'s1 shape:',s1.shape,'s2 shape:',s2.shape,'s3 shape:',s3.shape,'s4 shape:',s4.shape,'s5 feature',s5.shape, down5.shape)
        
        features = down5 #[2,4,11,21]
        #print(features.shape)
        #aaa

        classifier1 = self.up1(features)
        classifier1 = skipconnection(classifier1,s5) #up1 -> s4
        classifier1 = self.e1(classifier1)
        
        #aaaaaa
        
        classifier2 = self.up2(classifier1)
        #print(classifier2.shape)
        classifier2 = skipconnection(classifier2, s4)
        #print(classifier2.shape) #up2 -> s3
        classifier2 = self.e2(classifier2)
        

        classifier3 = self.up3(classifier2)
        classifier3 = skipconnection(classifier3, s3) #up3 -> s2
        classifier3 = self.e3(classifier3)

        classifier4 = self.up4(classifier3)
        classifier4 = skipconnection(classifier4, s2) #up4 -> s1
        classifier4 = self.e4(classifier4)
        #print('classifier size:',classifier4.shape)
        #aaa
        classifier5 = self.up5(classifier4)
        classifier5 = skipconnection(classifier5, s1)
        classifier5 = self.e5(classifier5)

        classifier6 = self.up6(classifier5)
        classifier6 = skipconnection(classifier6, s0)
        classifier6 = self.e6(classifier6)

        #print('classifier size:',classifier6.shape)
        #aaa

        xt_classifier = self.outc(classifier6)  # [B, 4, H, W]
        #print('classifier shape:', xt_classifier.shape)
        pred = nn.functional.interpolate(xt_classifier, size=(h, w), mode='bilinear', align_corners=True)
        #print('pred shape:',pred.shape)
        #print('features shape', features.shape)
        
        return features, xt_classifier, pred


    def forward_D(self, features):
        #flatten the features
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        conv1 = nn.Conv2d(256, 64, 1).to(device)
        features = conv1(features)
        features = features.view(features.size(0), -1)
        #print(features.shape)
        
        logit = self.D(features)
        return logit    # real/syth
######################################################################
    #adoptimize
    def regloss(self):
        Dregloss = [torch.sum(layer.weight ** 2) / 2 for layer in self.D if type(layer) == nn.Linear]
        layers = chain(self.init_encoder, self.s1, self.s2, self.s3,self.s4,self.s5)
        Gregloss = [torch.sum(layer.weight ** 2) / 2 for layer in layers if type(layer) == nn.Conv2d or type(layer) == nn.Linear]
        mean = lambda x:0.0005 * torch.mean(torch.stack(x))
        return mean(Dregloss), mean(Gregloss)

    def adloss(self, s_logits, t_logits): #loss of adersarial
        # sigmoid binary cross entropy with reduce mean
        #print(s_logits.shape, src_domain_lbl.shape)
        #print(s_logits,src_domain_lbl)
        D_real_loss = self.BCEloss(t_logits.detach(), torch.ones_like(t_logits))
        D_fake_loss = self.BCEloss(s_logits.detach(), torch.zeros_like(s_logits))
        D_loss = (D_real_loss + D_fake_loss) * 0.1 ## alpha to weight the influence of domain adaptation
        G_loss = -D_loss

        return D_loss, G_loss
    #smloss
    def smloss(self, s_feature, t_feature, y_s, y_t): #loss of domain adaptation(SMloss)

        n, c, h, w = s_feature.shape
        np, cp, hp, wp = y_t.shape
        #print(s_feature.shape, y_t.shape)
        
        #t_labels = 
        # get labels
        s_labels, t_labels = y_s, torch.argmax(y_t, dim=1) # torch.Size([2, 376, 672])

        # image number in each class 132-160
        ones = torch.ones_like(s_labels, dtype=torch.float, device=s_labels.device)
        zeros = torch.zeros(self.n_class, hp, wp, device=s_labels.device)

        s_n_classes = zeros.scatter_add(0, s_labels, ones)
        t_n_classes = zeros.scatter_add(0, t_labels, ones)
        #print(s_n_classes.shape)
        # image number cannot be 0, when calculating centroids
        ones = torch.ones_like(s_n_classes)
        s_n_classes = torch.max(s_n_classes, ones)
        t_n_classes = torch.max(t_n_classes, ones)
        #print('max size',s_n_classes.shape)

        s_n_classes = torch.mean(s_n_classes.float(), dim=[1,2], keepdim=True)
        t_n_classes = torch.mean(t_n_classes.float(), dim=[1,2], keepdim=True)
        #print('max size',s_n_classes.shape) #%%%%% 
        
        # calculating centroids, sum and divide
        zeros = torch.zeros(self.n_class, c, h, w, device=s_labels.device)
        #print(zeros.shape)
        #print("s_labels shape:", s_labels.shape)
        #print("t_labels shape:", t_labels.shape)

        s_labels = torch.mean(s_labels.float(), dim=[1,2], keepdim=True)
        t_labels = torch.mean(t_labels.float(), dim=[1,2], keepdim=True)
        s_labels = torch.Tensor(s_labels).long()
        t_labels = torch.Tensor(t_labels).long()

        #print("s_labels shape:", s_labels.shape)
        #print("t_labels shape:", t_labels.shape)
        #aaa

        s_sum_feature = zeros.scatter_add_(0, s_labels.unsqueeze(1).expand(2, c, 5, 10), s_feature)
        t_sum_feature = zeros.scatter_add_(0, t_labels.unsqueeze(1).expand(2, c, 5, 10), t_feature)
        current_s_centroid = torch.div(s_sum_feature, s_n_classes.view(self.n_class, 1,1,1))
        current_t_centroid = torch.div(t_sum_feature, t_n_classes.view(self.n_class, 1,1,1))
        #print(current_s_centroid.shape, current_t_centroid.shape)
        
        # Moving Centroid(mean)
        #print(self.s_centroid.shape,self.t_centroid.shape)
        #aaa
        s_centroid = (1-0.3) * self.s_centroid + 0.3 * current_s_centroid
        t_centroid = (1-0.3) * self.t_centroid + 0.3 * current_t_centroid
        semantic_loss = self.MSEloss(s_centroid, t_centroid) # norm-2
        #aaa
        self.s_centroid = s_centroid.detach()
        self.t_centroid = t_centroid.detach()

        return semantic_loss


    def c_loss(self, y_pred, y, weights):  #loss of classification
        C_loss = F.cross_entropy(y_pred, y, weight=weights)
        return C_loss

#169-223
    def get_optimizer(self, init_lr, lr_mult, lr_mult_D):
        w_finetune, b_finetune, w_train, b_train, w_D, b_D = [], [], [], [], [], []

        finetune_layers = chain(self.init_encoder,self.s1,self.s2, self.s3,self.s4,self.s5, self.outc)
        train_layers = chain(self.e1, self.e2,self.e3, self.e4, self.e5, self.e6)
        
        for layer in finetune_layers:
            if type(layer) == nn.Conv2d or type(layer) == nn.Linear or type(layer) == nn.ConvTranspose2d:
                w_finetune.append(layer.weight)
                #w_finetune = [param for param in w_finetune if param is not None]
                b_finetune.append(layer.bias)
        
        for layer in train_layers:
            if type(layer) == nn.Conv2d or type(layer) == nn.Linear or type(layer) == nn.ConvTranspose2d:
                w_train.append(layer.weight)
                #w_train = [param for param in w_train if param is not None]
                b_train.append(layer.bias)
                #b_train = [param for param in b_train if param is not None]
        for layer in self.D:
            if type(layer) == nn.Linear:
                w_D.append(layer.weight)
                b_D.append(layer.bias)
        #print(w_train, b_train)
        w_finetune = [param for param in w_finetune if param is not None]
        b_finetune = [param for param in b_finetune if param is not None]
        w_train = [param for param in w_train if param is not None]
        b_train = [param for param in b_train if param is not None]
        opt = torch.optim.SGD([{'params': w_finetune, 'lr': init_lr * lr_mult[0]},
                               {'params': b_finetune, 'lr': init_lr * lr_mult[1]},
                               {'params': w_train, 'lr': init_lr * lr_mult[2]},
                               {'params': b_train, 'lr': init_lr * lr_mult[3]}],
                              lr=init_lr,momentum=0.9)

        opt_D = torch.optim.SGD([{'params': w_D, 'lr': init_lr * lr_mult_D[0]},
                                 {'params': b_D, 'lr': init_lr * lr_mult_D[1]}],
                                lr=init_lr,momentum=0.9)

        return opt, opt_D

##############################################################
class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                                        stride=1,
                                        padding=(int((local_size - 1.0) / 2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size,
                                        stride=1,
                                        padding=int((local_size - 1.0) / 2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x

def truncated_normal_(tensor, mean=0, std=0.01):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)