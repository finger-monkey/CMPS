from __future__ import print_function, absolute_import
import os.path as osp
import numpy as np
from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import read_json
from ..utils.serialization import write_json


def _pluck(identities, indices, relabel=False):
    ret = []
    for index, pid in enumerate(indices):
        pid_images = identities[pid]
        for camid, cam_images in enumerate(pid_images):
            for fname in cam_images:
                name = osp.splitext(fname)[0]
                x, y, _ = map(int, name.split('_'))
                assert pid == x and camid == y
                if relabel:
                    ret.append((fname, index, camid))
                else:
                    ret.append((fname, pid, camid))
    return ret

class Sysu(Dataset):
    url = 'https://drive.google.com/file/5fh-rUzbwVRfdgs453ytzWG9COHM/view'
    md5 = '34005ab7d134rgs4eeafe81gr433'

    def __init__(self, root, split_id=0, num_val=96, download=True):
        super(Sysu, self).__init__(root, split_id=split_id)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "You can use download=True to download it.")

        self.load(num_val)

    def download(self):
        if self._check_integrity():
            return

        import re
        import hashlib
        import shutil
        from glob import glob
        from zipfile import ZipFile

        raw_dir = osp.join(self.root, 'raw')
        mkdir_if_missing(raw_dir)

        # Download the raw zip file
        fpath = osp.join(raw_dir, 'sysu_v2.zip')

        if osp.isfile(fpath):
            print("Using downloaded file: " + fpath)
        else:
            raise RuntimeError("Please download the dataset manually from {} "
                               "to {}".format(self.url, fpath))

        # Extract the file
        exdir = raw_dir
        if not osp.isdir(exdir):
            print("Extracting zip file")
            with ZipFile(fpath) as z:
                z.extractall(path=raw_dir)

        # Format
        images_dir = osp.join(self.root, 'images')
        mkdir_if_missing(images_dir)

        # XX identities (+1 for background) with X camera views each
        identities = [[[] for _ in range(6)] for _ in range(534)]
        # print("identities=",identities)

        def register(subdir, pattern=re.compile(r'([-\d]+)_c(\d)')):
            fnames = []
            fpaths = sorted(glob(osp.join(exdir, subdir, '*.jpg'))) #
            pids = set()
            for fpath in fpaths:
                fname = osp.basename(fpath)
                pid, cam = map(int, pattern.search(fname).groups())
                if pid == -1: continue  #
                assert 0 <= pid <= 534  #
                assert 1 <= cam <= 6
                cam -= 1
                pids.add(pid)
                fname = ('{:08d}_{:02d}_{:04d}.jpg'
                         .format(pid, cam, len(identities[pid][cam])))
                identities[pid][cam].append(fname)
                shutil.copy(fpath, osp.join(images_dir, fname))
                fnames.append(fname)
            return pids, fnames

        trainval_pids, _ = register('bounding_box_train')
        gallery_pids, gallery_fnames = register('bounding_box_test')
        query_pids, query_fnames = register('query')
        assert query_pids <= gallery_pids

        # Save meta information into a json file
        meta = {'name': 'SYSU', 'shot': 'multiple', 'num_cameras': 6,
                'identities': identities,
                'query_fnames': query_fnames,
                'gallery_fnames': gallery_fnames}
        write_json(meta, osp.join(self.root, 'meta.json'))

        # Save the only training / test split
        splits = [{
            'trainval': sorted(list(trainval_pids)),
            'query': sorted(list(query_pids)),
            'gallery': sorted(list(gallery_pids))}]
        write_json(splits, osp.join(self.root, 'splits.json'))

    def load(self, num_val=0.3, verbose=True):
        splits = read_json(osp.join(self.root, 'splits.json'))
        if self.split_id >= len(splits):
            raise ValueError("split_id exceeds total splits {}"
                             .format(len(splits)))
        self.split = splits[self.split_id]

        # Randomly split train / val
        trainval_pids = np.asarray(self.split['trainval'])
        np.random.shuffle(trainval_pids)
        num = len(trainval_pids)
        if isinstance(num_val, float):
            num_val = int(round(num * num_val))
        if num_val >= num or num_val < 0:
            raise ValueError("num_val exceeds total identities {}"
                             .format(num))
        train_pids = sorted(trainval_pids[:-num_val])
        val_pids = sorted(trainval_pids[-num_val:])

        self.meta = read_json(osp.join(self.root, 'meta.json'))
        identities = self.meta['identities']

        self.train = _pluck(identities, train_pids, relabel=True)
        self.val = _pluck(identities, val_pids, relabel=True)
        self.trainval = _pluck(identities, trainval_pids, relabel=True)
        self.num_train_ids = len(train_pids)
        self.num_val_ids = len(val_pids)
        self.num_trainval_ids = len(trainval_pids)

        query_fnames = self.meta['query_fnames']
        gallery_fnames = self.meta['gallery_fnames']
        self.query = []
        for fname in query_fnames:
            name = osp.splitext(fname)[0]
            pid, cam, _ = map(int, name.split('_'))
            self.query.append((fname, pid, cam))
        self.gallery = []
        for fname in gallery_fnames:
            name = osp.splitext(fname)[0]
            pid, cam, _ = map(int, name.split('_'))
            self.gallery.append((fname, pid, cam))

        if verbose:
            print(self.__class__.__name__, "dataset loaded")
            print("  Subset   | # ids | # images")
            print("  ---------------------------")
            print("  Train    | {:5d} | {:8d}"
                  .format(self.num_train_ids, len(self.train)))
            print("  Val      | {:5d} | {:8d}"
                  .format(self.num_val_ids, len(self.val)))
            print("  Trainval | {:5d} | {:8d}"
                  .format(self.num_trainval_ids, len(self.trainval)))
            print("  Query    | {:5d} | {:8d}"
                  .format(len(self.split['query']), len(self.query)))
            print("  Gallery  | {:5d} | {:8d}"
                  .format(len(self.split['gallery']), len(self.gallery)))


