from __future__ import print_function, absolute_import  
import time
from collections import OrderedDict
from collections import namedtuple
import torch
from .evaluation_metrics import cmc, mean_ap
from .feature_extraction import extract_cnn_feature, extract_pcb_feature
from .utils.meters import AverageMeter

def extract_features(model, data_loader, print_freq=1, metric=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    features = OrderedDict()
    labels = OrderedDict()
    end = time.time()
    for i, (imgs, fnames, pids, _) in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs = extract_cnn_feature(model, imgs)
        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output
            labels[fname] = pid
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t\t'
                  'Batch Loader Time {:.3f} ({:.3f})\t\t'
                  'Data Loader Time {:.3f} ({:.3f})\t\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))
    return features, labels
def pairwise_distance(features, query=None, gallery=None, metric=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist = dist.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist
def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 10,20)):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.2%}'.format(mAP))
    cmc_configs = {
        'score': dict(separate_camera_set=False,
                       single_gallery_shot=False,
                       first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}
    print('CMC Scores{:>12}'
          .format('score'))

    rank_score = namedtuple(
        'rank_score',
        ['map', 'score'],
    )
    for k in cmc_topk:
        print('  top-{:<4}{:12.2%}'
              .format(k,
                      cmc_scores['score'][k - 1]))
    score = rank_score(
        mAP,
        cmc_scores['score'],
    )
    return score
class Evaluator(object):
    def __init__(self, model, print_freq=1):
        super(Evaluator, self).__init__()
        self.model = model
        self.print_freq = print_freq

    def evaluate(self, data_loader, query, gallery, metric=None):
        features, _ = extract_features(self.model, data_loader, print_freq=self.print_freq)
        distmat = pairwise_distance(features, query, gallery, metric=metric)
        return evaluate_all(distmat, query=query, gallery=gallery)

    def evaMat(self, distMat, query, gallery, saveRank=False, root=None):
        return evaluate_all(distMat, query=query, gallery=gallery)
