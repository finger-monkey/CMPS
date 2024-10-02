import os
import os.path as osp
import argparse
import sys
import torch
from torch.utils.data import DataLoader
from reid import models
from torch.nn import functional as F
from reid import datasets
from MI_SGD import MI_SGD,keepGradUpdate
from test import test
from reid.utils.data import transforms as T
from torchvision.transforms import Resize
from reid.utils.data.preprocessor import Preprocessor
from reid.evaluators import Evaluator
from torch.optim.optimizer import Optimizer, required
import random
import numpy as np
import math
from reid.evaluators import extract_features
from reid.utils.meters import AverageMeter
import torchvision
import faiss
from torchvision import transforms
import logging
    
def get_data(sourceName, split_id, data_dir, height, width,
             batch_size, workers, combine):
    root = osp.join(data_dir, sourceName)

    sourceSet = datasets.create(sourceName, root, num_val=0.1, split_id=split_id)
    num_classes = sourceSet.num_trainval_ids if combine else sourceSet.num_train_ids
    tgtSet = sourceSet
    class_tgt = tgtSet.num_trainval_ids if combine else tgtSet.num_train_ids

    train_transformer = T.Compose([
        Resize((height, width)),
        transforms.RandomGrayscale(p=0.2),
        T.ToTensor(),
    ])

    train_transformer2 = T.Compose([
        Resize((height, width)),
        T.ToTensor(),
    ])

    train_step1 = DataLoader(
        Preprocessor(sourceSet.trainval, root=sourceSet.images_dir, transform=train_transformer),
        batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)
    train_step3 = DataLoader(
        Preprocessor(sourceSet.trainval, root=sourceSet.images_dir, transform=train_transformer2),
        batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)

    return sourceSet, sourceSet, num_classes, class_tgt, train_step1, train_step3




def calDist(qFeat, gFeat):
    m, n = qFeat.size(0), gFeat.size(0)
    x = qFeat.view(m, -1)
    y = gFeat.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
             torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m

