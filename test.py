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


MODE = "bilinear"

def test(dataset, net, noise, args, evaluator, epoch):
    print(">> Evaluating network on test datasets...")

    net = net.cuda()
    net.eval()
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    def add_noise(img):
        n = noise.cpu()
        img = img.cpu()
        n = F.interpolate(
            n.unsqueeze(0), mode=MODE, size=tuple(img.shape[-2:]), align_corners=True
        ).squeeze()
        return torch.clamp(img + n, 0, 1)

    query_trans = T.Compose([
        T.RectScale(args.height, args.width),
        T.ToTensor(), T.Lambda(lambda img: add_noise(img)),
        normalize
    ])
    test_transformer = T.Compose([
        T.RectScale(args.height, args.width),
        T.ToTensor(), normalize
    ])
    query_loader = DataLoader(
        Preprocessor(dataset.query, root=dataset.images_dir, transform=query_trans),
        batch_size=args.batch_size, num_workers=0, shuffle=False, pin_memory=True
    )
    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery, root=dataset.images_dir, transform=test_transformer),
        batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True
    )
    qFeats, gFeats, qnames, gnames = [], [], [], []
    with torch.no_grad():
        for (inputs, qname, _, _) in query_loader:
            inputs = inputs.cuda()
            qFeats.append(net(inputs)[0])
            qnames.extend(qname)
        qFeats = torch.cat(qFeats, 0)
        for (inputs, gname, _, _) in gallery_loader:
            inputs = inputs.cuda()
            gFeats.append(net(inputs)[0])
            gnames.extend(gname)
        gFeats = torch.cat(gFeats, 0)
    distMat = calDist(qFeats, gFeats)

    # evaluate on test datasets
    evaluator.evaMat(distMat, dataset.query, dataset.gallery)
    return

def calDist(qFeat, gFeat):
    m, n = qFeat.size(0), gFeat.size(0)
    x = qFeat.view(m, -1)
    y = gFeat.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
             torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m
