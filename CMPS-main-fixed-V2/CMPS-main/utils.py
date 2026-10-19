import os.path as osp
import torch
from torch.utils.data import DataLoader
from reid import datasets
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from torchvision import transforms


def get_data(sourceName, split_id, data_dir, height, width, batch_size, workers, combine):
    root = osp.join(data_dir, sourceName)
    sourceSet = datasets.create(sourceName, root, num_val=0.1, split_id=split_id)
    num_classes = sourceSet.num_trainval_ids if combine else sourceSet.num_train_ids
    class_tgt = sourceSet.num_trainval_ids if combine else sourceSet.num_train_ids

    visible_cameras = set(getattr(sourceSet, 'visible_cameras', ()))
    infrared_cameras = set(getattr(sourceSet, 'infrared_cameras', ()))
    if not visible_cameras or not infrared_cameras:
        raise ValueError('Dataset must define visible_cameras and infrared_cameras.')
    mapped_cameras = visible_cameras | infrared_cameras
    unmapped_cameras = sorted({item[2] for item in sourceSet.trainval if item[2] not in mapped_cameras})
    if unmapped_cameras:
        raise ValueError('Dataset {} has unmapped training camera IDs: {}'.format(sourceName, unmapped_cameras))

    visible_train = [item for item in sourceSet.trainval if item[2] in visible_cameras]
    infrared_train = [item for item in sourceSet.trainval if item[2] in infrared_cameras]
    if not visible_train or not infrared_train:
        camera_ids = sorted(set(item[2] for item in sourceSet.trainval))
        raise ValueError(
            'Could not build both modality streams for {}. Found camera IDs {}; visible IDs {}; infrared IDs {}.'.format(
                sourceName, camera_ids, sorted(visible_cameras), sorted(infrared_cameras)
            )
        )

    train_transformer = T.Compose([
        transforms.Resize((height, width)),
        transforms.RandomGrayscale(p=0.2),
        T.ToTensor(),
    ])
    center_transformer = T.Compose([
        transforms.Resize((height, width)),
        T.ToTensor(),
    ])
    gray_transformer = T.Compose([
        transforms.Resize((height, width)),
        transforms.Grayscale(num_output_channels=3),
        T.ToTensor(),
    ])

    train_step1 = DataLoader(
        Preprocessor(visible_train, root=sourceSet.images_dir, transform=train_transformer),
        batch_size=batch_size, num_workers=workers, shuffle=True, pin_memory=True
    )
    train_step3 = DataLoader(
        Preprocessor(infrared_train, root=sourceSet.images_dir, transform=train_transformer),
        batch_size=batch_size, num_workers=workers, shuffle=True, pin_memory=True
    )
    center_visible_loader = DataLoader(
        Preprocessor(visible_train, root=sourceSet.images_dir, transform=center_transformer),
        batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True
    )
    center_infrared_loader = DataLoader(
        Preprocessor(infrared_train, root=sourceSet.images_dir, transform=center_transformer),
        batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True
    )
    center_gray_visible_loader = DataLoader(
        Preprocessor(visible_train, root=sourceSet.images_dir, transform=gray_transformer),
        batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True
    )
    center_gray_infrared_loader = DataLoader(
        Preprocessor(infrared_train, root=sourceSet.images_dir, transform=gray_transformer),
        batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True
    )

    return (
        sourceSet,
        sourceSet,
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
    )


def calDist(qFeat, gFeat):
    m, n = qFeat.size(0), gFeat.size(0)
    x = qFeat.view(m, -1)
    y = gFeat.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
             torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(x, y.t(), beta=1, alpha=-2)
    return dist_m
