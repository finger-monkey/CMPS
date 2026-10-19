import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable
import pdb


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()
    def forward(self, pred, label):
        # pred: 2D matrix (batch_size, num_classes)
        # label: 1D vector indicating class number
        T=3

        predict = F.log_softmax(pred/T,dim=1)
        target_data = F.softmax(label/T,dim=1)
        target_data =target_data+10**(-7)
        target = Variable(target_data.data.cuda(),requires_grad=False)
        loss=T*T*((target*(target.log()-predict)).sum(1).sum()/target.size()[0])
        return loss

class OriTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    
    def __init__(self, batch_size, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct


class TDRLoss(nn.Module):
    """Tri-directional ranking loss.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(TDRLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.shape[0] // 3
        input1 = inputs.narrow(0, 0, n)
        input2 = inputs.narrow(0, n, n)
        input3 = inputs.narrow(0, 2 * n, n)

        dist1 = pdist_torch(input1, input2)
        dist2 = pdist_torch(input2, input3)
        dist3 = pdist_torch(input1, input3)

        # compute mask
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        # P: 1 2 N: 3
        dist_ap1, dist_an1 = [], []
        for i in range(n):
            dist_ap1.append(dist1[i][mask[i]].max().unsqueeze(0))
            dist_an1.append(dist3[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap1 = torch.cat(dist_ap1)
        dist_an1 = torch.cat(dist_an1)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an1)
        loss1 = self.ranking_loss(dist_an1, dist_ap1, y)

        # P: 2 3 N: 1
        dist_ap2, dist_an2 = [], []
        for i in range(n):
            dist_ap2.append(dist2[i][mask[i]].max().unsqueeze(0))
            dist_an2.append(dist1[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap2= torch.cat(dist_ap2)
        dist_an2 = torch.cat(dist_an2)

        # Compute ranking hinge loss
        loss2 = self.ranking_loss(dist_an2, dist_ap2, y)

        # P: 3 1 N: 2
        dist_ap3, dist_an3 = [], []
        for i in range(n):
            dist_ap3.append(dist3[i][mask[i]].max().unsqueeze(0))
            dist_an3.append(dist2[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap3 = torch.cat(dist_ap3)
        dist_an3 = torch.cat(dist_an3)

        # Compute ranking hinge loss
        loss3 = self.ranking_loss(dist_an3, dist_ap3, y)

        # compute accuracy
        correct1 = torch.ge(dist_an1, dist_ap1).sum().item()
        correct2 = torch.ge(dist_an2, dist_ap2).sum().item()
        correct3 = torch.ge(dist_an3, dist_ap3).sum().item()


        # regularizer
        # pdb.set_trace()
        loss_reg = dist_ap1.mean() + dist_ap2.mean() + dist_ap3.mean()
        return loss1+loss2+loss3, loss_reg, correct1 + correct2+correct3

class WTDRLoss(nn.Module):
    """Tri-directional ranking loss.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(WTDRLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(reduction='none', margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.shape[0] // 3
        input1 = inputs.narrow(0, 0, n)
        input2 = inputs.narrow(0, n, n)
        input3 = inputs.narrow(0, 2 * n, n)

        dist1 = pdist_torch(input1, input2)
        dist2 = pdist_torch(input2, input3)
        dist3 = pdist_torch(input1, input3)

        # compute mask
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        # P: 1 2 N: 3
        dist_ap1, dist_an1 = [], []
        for i in range(n):
            dist_ap1.append(dist1[i][mask[i]].max().unsqueeze(0))
            dist_an1.append(dist3[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap1 = torch.cat(dist_ap1)
        dist_an1 = torch.cat(dist_an1)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an1)
        loss1 = self.ranking_loss(dist_an1, dist_ap1, y)
        weights1 = loss1.data.exp()
        # weights1 = loss1.data.pow(2)

        # P: 2 3 N: 1
        dist_ap2, dist_an2 = [], []
        for i in range(n):
            dist_ap2.append(dist2[i][mask[i]].max().unsqueeze(0))
            dist_an2.append(dist1[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap2= torch.cat(dist_ap2)
        dist_an2 = torch.cat(dist_an2)

        # Compute ranking hinge loss
        loss2 = self.ranking_loss(dist_an2, dist_ap2, y)
        weights2 = loss2.data.exp()
        # weights2 = loss2.data.pow(2)


        # P: 3 1 N: 2
        dist_ap3, dist_an3 = [], []
        for i in range(n):
            dist_ap3.append(dist3[i][mask[i]].max().unsqueeze(0))
            dist_an3.append(dist2[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap3 = torch.cat(dist_ap3)
        dist_an3 = torch.cat(dist_an3)

        # Compute ranking hinge loss
        loss3 = self.ranking_loss(dist_an3, dist_ap3, y)
        weights3 = loss3.data.exp()
        # weights3 = loss3.data.pow(2)

        # compute accuracy
        correct1 = torch.ge(dist_an1, dist_ap1).sum().item()
        correct2 = torch.ge(dist_an2, dist_ap2).sum().item()
        correct3 = torch.ge(dist_an3, dist_ap3).sum().item()


        # weighted aggregation loss
        weights_sum = torch.cat((weights1, weights2, weights3),0)
        wloss1 = torch.mul(weights1.div_(weights_sum.sum()), loss1).sum()
        wloss2 = torch.mul(weights2.div_(weights_sum.sum()), loss2).sum()
        wloss3 = torch.mul(weights3.div_(weights_sum.sum()), loss3).sum()

        return 3*(wloss1+wloss2+wloss3), correct1 + correct2+correct3

        
class BDTRLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.suffix
    """
    def __init__(self, batch_size, margin=0.5):
        super(BDTRLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.batch_size = batch_size
        self.mask = torch.eye(batch_size)
    def forward(self, input, target):
        """
        Args:
        - input: feature matrix with shape (batch_size, feat_dim)
        - target: ground truth labels with shape (num_classes)
        """
        n = self.batch_size
        input1 = input.narrow(0,0,n)
        input2 = input.narrow(0,n,n)

        # modal 1 to modal 2
        # Compute modal 1 to modal 2 distance
        dist = pdist_torch(input1, input2)

        dist_ap1, dist_an1 = [], []
        for i in range(n):
            dist_ap1.append(dist[i,i].unsqueeze(0))
            dist_an1.append(dist[i][self.mask[i] == 0].min().unsqueeze(0))
        dist_ap1 = torch.cat(dist_ap1)
        dist_an1 = torch.cat(dist_an1)
        
        # Compute ranking hinge loss for modal 1 to modal 2
        y = torch.ones_like(dist_an1)
        loss1 = self.ranking_loss(dist_an1, dist_ap1, y)
        
        # compute accuracy
        correct1  =  torch.ge(dist_an1, dist_ap1).sum().item()

        # modal 2 to modal 1
        # Compute modal 1 to modal 2 distance
        dist2 = pdist_torch(input2, input1)
        
        # For each anchor, find the hardest positive and negative
        dist_ap2, dist_an2 = [], []
        for i in range(n):
            dist_ap2.append(dist2[i,i].unsqueeze(0))
            dist_an2.append(dist2[i][self.mask[i] == 0].min().unsqueeze(0))
        dist_ap2 = torch.cat(dist_ap2)
        dist_an2 = torch.cat(dist_an2)
        
        # Compute ranking hinge loss for modal 2 to modal 1
        y2 = torch.ones_like(dist_an2)
        loss2 = self.ranking_loss(dist_an2, dist_ap2, y2)
        
        # compute accuracy
        correct2  =  torch.ge(dist_an2, dist_ap2).sum().item()
        
        inter_loss = torch.add(loss1, loss2)

        # computer intra-modality loss


        return inter_loss, correct1 + correct2

class CTriLoss:
    def __init__(self, rho):
        self.rho = rho

    def __call__(self, C_n_g, f_adv_RGB, C_p_ir, f_adv_RGB_ir, C_n_RGB, f_adv_ir, C_p_g, f_adv_ir_g, C_n_ir, f_adv_g, C_p_RGB, f_adv_g_RGB):
        loss1 = np.maximum(np.linalg.norm(C_n_g - f_adv_RGB) - np.linalg.norm(C_p_ir - f_adv_RGB_ir) + self.rho, 0)
        loss2 = np.maximum(np.linalg.norm(C_n_RGB - f_adv_ir) - np.linalg.norm(C_p_g - f_adv_ir_g) + self.rho, 0)
        loss3 = np.maximum(np.linalg.norm(C_n_ir - f_adv_g) - np.linalg.norm(C_p_RGB - f_adv_g_RGB) + self.rho, 0)

        total_loss = loss1 + loss2 + loss3
        return total_loss


def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx    


def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis = 1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis = 1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    # dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx