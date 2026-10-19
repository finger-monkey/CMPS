import os
import argparse
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from reid import models, datasets
from MI_SGD import MI_SGD, keepGradUpdate
from test import test
from utils import get_data
from reid.utils.data import transforms as T
from reid.evaluators import Evaluator
from reid.utils.meters import AverageMeter
import random
import numpy as np
import math
import faiss
import time


def _paired_batches(loader_a, loader_b):
    if len(loader_a) == 0 or len(loader_b) == 0:
        raise ValueError('Both modality data loaders must contain at least one batch.')
    iterator_a = iter(loader_a)
    iterator_b = iter(loader_b)
    for _ in range(max(len(loader_a), len(loader_b))):
        try:
            batch_a = next(iterator_a)
        except StopIteration:
            iterator_a = iter(loader_a)
            batch_a = next(iterator_a)
        try:
            batch_b = next(iterator_b)
        except StopIteration:
            iterator_b = iter(loader_b)
            batch_b = next(iterator_b)
        yield batch_a, batch_b


def _forward_features(net, inputs, modal=1):
    if hasattr(net, 'visible_module') and hasattr(net, 'thermal_module'):
        output = net(inputs, modal=modal)
    else:
        output = net(inputs)
    return output[0] if isinstance(output, (tuple, list)) else output


def _feature_tensor(net, loader, samples, modal=1):
    if not samples:
        raise ValueError('Cannot extract features from an empty modality split.')
    net.eval()
    features = {}
    with torch.no_grad():
        for images, fnames, _, _ in loader:
            images = images.cuda(non_blocking=True)
            outputs = _forward_features(net, images, modal=modal).detach().cpu()
            for fname, output in zip(fnames, outputs):
                features[fname] = output
    return torch.stack([features[fname] for fname, _, _ in samples])


def _cluster_centroids(features, num_centroids, device):
    features = features.detach().cpu().numpy().astype(np.float32, copy=False)
    num_centroids = min(max(1, int(num_centroids)), len(features))
    if len(features) == 0:
        raise ValueError('Cannot cluster an empty feature tensor.')
    use_gpu = hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0
    cluster = faiss.Kmeans(features.shape[1], num_centroids, niter=20, gpu=use_gpu)
    cluster.train(features)
    return torch.from_numpy(cluster.centroids.copy()).to(device=device, dtype=torch.float32)


def _nearest_farthest(features, centroids):
    distances = torch.cdist(features.detach(), centroids, p=2)
    nearest = centroids[distances.argmin(dim=1)]
    farthest = centroids[distances.argmax(dim=1)]
    return nearest, farthest


def train_CMPS(train_step1_loader, train_step3_loader, net, noise, epoch, optimizer,
               rgb_centroids, infrared_centroids, gray_centroids, normalize):
    global args
    noise.requires_grad_(True)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    device = noise.device
    mean = torch.as_tensor(normalize.mean, device=device).view(1, 3, 1, 1)
    std = torch.as_tensor(normalize.std, device=device).view(1, 3, 1, 1)
    net.eval()
    end = time.time()
    optimizer.zero_grad()
    optimizer.rescale()

    for i, ((input, _, _, _), (iTest, _, _, _)) in enumerate(
            _paired_batches(train_step1_loader, train_step3_loader)):
        data_time.update(time.time() - end)
        input = input.to(device, non_blocking=True)
        iTest = iTest.to(device, non_blocking=True)

        with torch.no_grad():
            normInput = (input - mean) / std
            feature = _forward_features(net, normInput, modal=1)
            nearest_ir, _ = _nearest_farthest(feature, infrared_centroids)
            _, farthest_gray = _nearest_farthest(feature, gray_centroids)

        current_noise = F.interpolate(
            noise.unsqueeze(0), mode=MODE, size=tuple(input.shape[-2:]), align_corners=True
        ).squeeze(0)
        perturbed_input = torch.clamp(input + current_noise, 0, 1)
        perturbed_input_norm = (perturbed_input - mean) / std
        perturbed_feature = _forward_features(net, perturbed_input_norm, modal=1)
        optimizer.zero_grad()
        loss_step1 = 10 * F.triplet_margin_loss(
            perturbed_feature, farthest_gray, nearest_ir, margin=0.5
        )

        grad = torch.autograd.grad(loss_step1, noise, create_graph=True)[0]
        noise_one_step = keepGradUpdate(noise, optimizer, grad, MAX_EPS)
        new_noise = F.interpolate(
            noise_one_step.unsqueeze(0), mode=MODE,
            size=tuple(iTest.shape[-2:]), align_corners=True
        ).squeeze(0)

        with torch.no_grad():
            norm_infrared = (iTest - mean) / std
            infrared_feature = _forward_features(net, norm_infrared, modal=2)
            nearest_gray, _ = _nearest_farthest(infrared_feature, gray_centroids)
            _, farthest_rgb = _nearest_farthest(infrared_feature, rgb_centroids)

        perturbed_infrared = torch.clamp(iTest + new_noise, 0, 1)
        perturbed_infrared_norm = (perturbed_infrared - mean) / std
        perturbed_infrared_feature = _forward_features(net, perturbed_infrared_norm, modal=2)
        loss_step3 = 10 * F.triplet_margin_loss(
            perturbed_infrared_feature, farthest_rgb, nearest_gray, margin=0.5
        )

        final_loss = loss_step1 + loss_step3
        final_loss.backward()
        optimizer.step()
        losses.update(final_loss.item())
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                '>> Train: [{0}][{1}/{2}]\t'
                'Batch Loader Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t\t'
                'Data Loader Time {data_time.val:.3f} ({data_time.avg:.3f})\t\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Noise l2: {noise:.4f}'.format(
                    epoch + 1,
                    i, max(len(train_step1_loader), len(train_step3_loader)),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    noise=noise.norm(),
                )
            )

    noise.requires_grad_(False)
    print('Train {}: Loss: {}'.format(epoch, losses.avg))
    return losses.avg, noise


MODE = 'bilinear'


def _checkpoint_state(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict):
        for key in ('net', 'state_dict', 'model_state_dict', 'model'):
            value = checkpoint.get(key)
            if isinstance(value, dict) and value:
                checkpoint = value
                break
    if not isinstance(checkpoint, dict):
        raise RuntimeError('Checkpoint does not contain a state dictionary: {}'.format(checkpoint_path))
    state = {}
    for key, value in checkpoint.items():
        if not torch.is_tensor(value):
            continue
        normalized_key = key
        while normalized_key.startswith('module.'):
            normalized_key = normalized_key[len('module.'):]
        state[normalized_key] = value
    if not state:
        raise RuntimeError('No tensor parameters found in checkpoint: {}'.format(checkpoint_path))
    return state


def load_model_checkpoint(net, checkpoint_path, arch):
    state = _checkpoint_state(checkpoint_path)
    current = net.state_dict()
    classifier_tokens = ('classifier', 'out_att')
    compatible = {
        key: value for key, value in state.items()
        if key in current and current[key].shape == value.shape
    }
    total_keys = [key for key in current if not any(token in key.lower() for token in classifier_tokens)]
    matched_keys = [key for key in compatible if not any(token in key.lower() for token in classifier_tokens)]
    key_coverage = len(matched_keys) / max(1, len(total_keys))
    numel_coverage = sum(current[key].numel() for key in matched_keys) / max(
        1, sum(current[key].numel() for key in total_keys)
    )
    architecture_markers = {
        'AGW': ('NL_2.', 'NL_3.'),
        'DDAG': ('attention_0.', 'wpa.'),
    }
    if arch in architecture_markers:
        markers = architecture_markers[arch]
        if not any(any(key.startswith(marker) for key in state) for marker in markers):
            raise RuntimeError(
                'Checkpoint architecture does not match {}. Expected model-specific parameters {}.'.format(
                    arch, markers
                )
            )
    if key_coverage < 0.5 or numel_coverage < 0.5:
        raise RuntimeError(
            'Checkpoint {} is incompatible with {}: matched {:.1%} of non-classifier parameter keys and {:.1%} of parameter elements.'.format(
                checkpoint_path, arch, key_coverage, numel_coverage
            )
        )
    missing, unexpected = net.load_state_dict(compatible, strict=False)
    print(
        'Loaded {} into {}: {:.1%} non-classifier key coverage, {:.1%} parameter coverage, {} missing keys, {} unexpected keys.'.format(
            checkpoint_path, arch, key_coverage, numel_coverage, len(missing), len(unexpected)
        )
    )
    return net


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data', type=str, required=True, help='path to reid dataset')
    parser.add_argument('-s', '--source', type=str, default='sysu_v2', choices=datasets.names())
    parser.add_argument('-a', '--arch', type=str, default='DDAG', choices=models.names())
    parser.add_argument('--target-arch', type=str, default=None, choices=models.names())
    parser.add_argument('--target-source', type=str, default=None, choices=datasets.names())
    parser.add_argument('--target-data', type=str, default=None)
    parser.add_argument('--target-split', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=16, required=True, help='number of examples/minibatch')
    parser.add_argument('--num_batches', type=int, required=False, help='number of batches (default entire dataset)')
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--target-resume', type=str, default='', metavar='PATH')
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--height', type=int, default=256, help='input height')
    parser.add_argument('--width', type=int, default=128, help='input width')
    parser.add_argument('--num-instances', type=int, default=8)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--max-eps', default=8, type=int, help='max eps')
    parser.add_argument('--eval-direction', type=str, default='both', choices=('auto', 'vis2ir', 'ir2vis', 'both'))
    parser.add_argument('--combine-trainval', action='store_true', help='train and val sets together for training')

    args = parser.parse_args()
    target_arch = args.target_arch or args.arch
    target_source = args.target_source or args.source
    target_data = args.target_data or args.data
    target_split = args.target_split if args.target_split is not None else args.split

    if not torch.cuda.is_available():
        raise RuntimeError('CMPS currently requires a CUDA-enabled PyTorch environment.')

    (
        sourceSet,
        _,
        num_classes,
        class_tgt,
        train_step1,
        train_step3,
        center_visible_loader,
        center_infrared_loader,
        center_gray_visible_loader,
        center_gray_infrared_loader,
        visible_train,
        infrared_train,
    ) = get_data(
        args.source,
        args.split,
        args.data,
        args.height,
        args.width,
        args.batch_size,
        8,
        args.combine_trainval,
    )

    if target_source == args.source and target_data == args.data and target_split == args.split:
        targetSet = sourceSet
    else:
        target_root = os.path.join(target_data, target_source)
        targetSet = datasets.create(target_source, target_root, num_val=0.1, split_id=target_split)
    target_num_classes = targetSet.num_trainval_ids if args.combine_trainval else targetSet.num_train_ids

    model = models.create(args.arch, pretrained=True, num_classes=num_classes)
    modelTest = models.create(target_arch, pretrained=True, num_classes=target_num_classes)

    if args.resume:
        load_model_checkpoint(model, args.resume, args.arch)
    else:
        print('No source checkpoint supplied; the source model will use its initialized pretrained weights.')

    same_target_configuration = (
        target_arch == args.arch
        and target_source == args.source
        and target_data == args.data
        and target_split == args.split
    )
    target_checkpoint = args.target_resume or (args.resume if same_target_configuration else '')
    if target_checkpoint:
        load_model_checkpoint(modelTest, target_checkpoint, target_arch)
    elif not same_target_configuration:
        raise ValueError('A separate --target-resume checkpoint is required when the target model or dataset differs from the source configuration.')

    model = model.cuda().eval()
    modelTest = modelTest.cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in modelTest.parameters():
        parameter.requires_grad_(False)

    device = torch.device('cuda')
    rgb_features = _feature_tensor(model, center_visible_loader, visible_train, modal=1)
    infrared_features = _feature_tensor(model, center_infrared_loader, infrared_train, modal=2)
    gray_visible_features = _feature_tensor(model, center_gray_visible_loader, visible_train, modal=1)
    gray_infrared_features = _feature_tensor(model, center_gray_infrared_loader, infrared_train, modal=2)
    gray_features = torch.cat((gray_visible_features, gray_infrared_features), dim=0)
    rgb_centroids = _cluster_centroids(rgb_features, sourceSet.num_trainval_ids, device)
    infrared_centroids = _cluster_centroids(infrared_features, sourceSet.num_trainval_ids, device)
    gray_centroids = _cluster_centroids(gray_features, sourceSet.num_trainval_ids, device)

    evaluator = Evaluator(modelTest, args.print_freq)
    noise = torch.zeros((3, args.height, args.width), device=device)
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    noise.requires_grad_(True)
    MAX_EPS = args.max_eps / 255.0
    optimizer = MI_SGD(
        [{'params': [noise], 'lr': MAX_EPS / 10, 'momentum': 1, 'sign': True}],
        max_eps=MAX_EPS,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=math.exp(-0.01))

    for epoch in range(args.epoch):
        scheduler.step()
        loss, noise = train_CMPS(
            train_step1,
            train_step3,
            model,
            noise,
            epoch,
            optimizer,
            rgb_centroids,
            infrared_centroids,
            gray_centroids,
            normalize,
        )
        if epoch % 5 == 0 or epoch == args.epoch - 1:
            test(targetSet, modelTest, noise, args, evaluator, epoch, direction=args.eval_direction)
