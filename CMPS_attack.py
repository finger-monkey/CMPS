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
from utils import get_data,calDist
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
import time

def train_CMPS(train_step1_loader, train_step3_loader, net, noise, epoch, optimizer,
              centroids, metaCentroids, normalize):
    global args
    noise.requires_grad = True
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mean = torch.Tensor(normalize.mean).view(1, 3, 1, 1).cuda()
    std = torch.Tensor(normalize.std).view(1, 3, 1, 1).cuda()
    net.eval()
    end = time.time()
    optimizer.zero_grad()
    optimizer.rescale()
    for i, ((input, _, pid, _), (iTest, _, _, _)) in enumerate(zip(train_step1_loader, train_step3_loader)):
        data_time.update(time.time() - end)
        model.zero_grad()
        input = input.cuda()
        iTest = iTest.cuda()
        with torch.no_grad():
            normInput = (input - mean) / std
            feature, realPred = net(normInput)
            scores = centroids.mm(F.normalize(feature.t(), p=2, dim=0))
            # scores = centroids.mm(feature.t())
            realLab = scores.max(0, keepdim=True)[1]
            _, ranks = torch.sort(scores, dim=0, descending=True)
            pos_i = ranks[0, :]
            neg_i = ranks[-1, :]
        neg_feature = centroids[neg_i, :]
        pos_feature = centroids[pos_i, :]

        current_noise = noise
        current_noise = F.interpolate(
            current_noise.unsqueeze(0),
            mode=MODE, size=tuple(input.shape[-2:]), align_corners=True,
        ).squeeze()
        perturted_input = torch.clamp(input + current_noise, 0, 1)
        perturted_input_norm = (perturted_input - mean) / std
        perturbed_feature = net(perturted_input_norm)[0]
        optimizer.zero_grad()
        loss_step1 = 10 * F.triplet_margin_loss(perturbed_feature, neg_feature, pos_feature, 0.5)


        loss_step1 = loss_step1.view(1)
        loss = loss_step1

        grad = torch.autograd.grad(loss, noise, create_graph=True)[0]
        noiseOneStep = keepGradUpdate(noise, optimizer, grad, MAX_EPS)

        newNoise = F.interpolate(
            noiseOneStep.unsqueeze(0), mode=MODE,
            size=tuple(iTest.shape[-2:]), align_corners=True,
        ).squeeze()

        with torch.no_grad():
            normMte = (iTest - mean) / std
            mteFeat = net(normMte)[0]
            scores = metaCentroids.mm(F.normalize(mteFeat.t(), p=2, dim=0))
            metaLab = scores.max(0, keepdim=True)[1]
            _, ranks = torch.sort(scores, dim=0, descending=True)
            pos_i = ranks[0, :]
            neg_i = ranks[-1, :]
        neg_mte_feat = metaCentroids[neg_i, :]
        pos_mte_feat = metaCentroids[pos_i, :]

        perMteInput = torch.clamp(iTest + newNoise, 0, 1)
        normPerMteInput = (perMteInput - mean) / std
        normMteFeat = net(normPerMteInput)[0]

        loss_step3 = 10 * F.triplet_margin_loss(
            normMteFeat, neg_mte_feat, pos_mte_feat, 0.5
        )

        finalLoss = loss_step3  + loss_step1
        finalLoss.backward()

        losses.update(loss_step1.item())
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                ">> Train: [{0}][{1}/{2}]\t"
                "Batch Loader Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t\t"
                "Data Loader Time {data_time.val:.3f} ({data_time.avg:.3f})\t\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Noise l2: {noise:.4f}".format(
                    epoch + 1,
                    i, len(train_step1_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses, loss_step3=loss_step3.item(),
                    noise=noise.norm(),
                )
            )
    noise.requires_grad = False
    print(f"Train {epoch}: Loss: {losses.avg}")
    return losses.avg, noise

MODE = "bilinear"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=''
    )

    parser.add_argument('--data', type=str, required=True,
                        help='path to reid dataset')
    parser.add_argument('-s', '--source', type=str, default='sysu',
                        choices=datasets.names())
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--batch_size', type=int, default=16, required=True,
                        help='number of examples/minibatch')
    parser.add_argument('--num_batches', type=int, required=False,
                        help='number of batches (default entire dataset)')
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--height', type=int, default=256,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--num-instances', type=int, default=8,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument("--max-eps", default=8, type=int, help="max eps")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")

    args = parser.parse_args()

    sourceSet, sourceSet, num_classes, class_tgt, train_step1, train_step3 = \
        get_data(args.source,
                 args.split, args.data, args.height,
                 args.width, args.batch_size, 8, args.combine_trainval)

    model = models.create(args.arch, pretrained=True, num_classes=num_classes)
    modelTest = models.create(args.arch, pretrained=True, num_classes=class_tgt)
    if args.resume:
        checkpoint = torch.load(args.resume)
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        try:
            model.load_state_dict(checkpoint)
        except:
            allNames = list(checkpoint.keys())
            for name in allNames:
                if name.count('classifier') != 0:
                    del checkpoint[name]
            model.load_state_dict(checkpoint, strict=False)
        #for test
        checkTgt = torch.load(args.resume)
        if 'state_dict' in checkTgt.keys():
            checkTgt = checkTgt['state_dict']
        try:
            modelTest.load_state_dict(checkTgt)
        except:
            allNames = list(checkTgt.keys())
            for name in allNames:
                if name.count('classifier') != 0:
                    del checkTgt[name]
            modelTest.load_state_dict(checkTgt, strict=False)

    model.eval()
    modelTest.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        modelTest = modelTest.cuda()
    features, _ = extract_features(model, train_step1, print_freq=10)
    features = torch.stack([features[f] for f, _, _ in sourceSet.trainval])
    metaFeats, _ = extract_features(model, train_step3, print_freq=10)
    metaFeats = torch.stack([metaFeats[f] for f, _, _ in sourceSet.trainval])
    if args.source == "sysu":
        ncentroids = 395
    else:
        ncentroids = 206
    fDim = features.shape[1]
    cluster, metaClu = faiss.Kmeans(fDim, ncentroids, niter=20, gpu=True), \
                       faiss.Kmeans(fDim, ncentroids, niter=20, gpu=True)
    cluster.train(features.cpu().numpy())
    metaClu.train(metaFeats.cpu().numpy())
    centroids = torch.from_numpy(cluster.centroids).cuda().float()
    metaCentroids = torch.from_numpy(metaClu.centroids).cuda().float()
    del metaClu, cluster
    evaluator = Evaluator(modelTest, args.print_freq)
    evaSrc = Evaluator(model, args.print_freq)
    noise = torch.zeros((3, args.height, args.width)).cuda()
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    noise.requires_grad = True
    MAX_EPS = args.max_eps / 255.0
    optimizer = MI_SGD(
        [{"params": [noise], "lr": MAX_EPS / 10, "momentum": 1, "sign": True}],
        max_eps=MAX_EPS,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=math.exp(-0.01))
    for epoch in range(args.epoch):
        scheduler.step()
        begin_time = time.time()
        loss, noise = train_CMPS(
            train_step1, train_step3, model, noise, epoch, optimizer,
            centroids, metaCentroids, normalize
        )
        if epoch % 5 == 0:
            test(sourceSet, modelTest, noise, args, evaluator, epoch)


