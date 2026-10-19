from __future__ import print_function, absolute_import
import os.path as osp
import numpy as np
from ..utils.data.dataset import Dataset, _pluck, _pluck_query, _pluck_gallery
from ..utils.serialization import read_json


class LLCM(Dataset):
    visible_cameras = (0, 1, 2, 3)
    infrared_cameras = (4, 5, 6, 7, 8)

    def __init__(self, root, split_id=0, num_val=0.1, download=False):
        super(LLCM, self).__init__(root, split_id=split_id)
        if not self._check_integrity():
            raise RuntimeError(
                'LLCM data must be prepared in the processed format with images/, meta.json and splits.json under {}'.format(root)
            )
        self.load(num_val)

    def load(self, num_val=0.1, verbose=True):
        splits = read_json(osp.join(self.root, 'splits.json'))
        if self.split_id >= len(splits):
            raise ValueError('split_id exceeds total splits {}'.format(len(splits)))
        self.split = splits[self.split_id]

        trainval_pids = np.asarray(self.split['trainval'])
        np.random.shuffle(trainval_pids)
        total = len(trainval_pids)
        if isinstance(num_val, float):
            num_val = int(round(total * num_val))
        if num_val < 0 or num_val >= total:
            raise ValueError('num_val must be in [0, {})'.format(total))
        if num_val == 0:
            train_pids = sorted(trainval_pids.tolist())
            val_pids = []
        else:
            train_pids = sorted(trainval_pids[:-num_val].tolist())
            val_pids = sorted(trainval_pids[-num_val:].tolist())

        self.meta = read_json(osp.join(self.root, 'meta.json'))
        identities = self.meta['identities']
        self.train = _pluck(identities, train_pids, relabel=True)
        self.val = _pluck(identities, val_pids, relabel=True)
        self.trainval = _pluck(identities, trainval_pids.tolist(), relabel=True)
        self.num_train_ids = len(train_pids)
        self.num_val_ids = len(val_pids)
        self.num_trainval_ids = total

        fname_lookup = {}
        for pid, camera_groups in enumerate(identities):
            for camid, fnames in enumerate(camera_groups):
                for fname in fnames:
                    fname_lookup[fname] = (fname, pid, camid)

        query_fnames = self.meta.get('query_fnames')
        gallery_fnames = self.meta.get('gallery_fnames')
        if query_fnames is not None and gallery_fnames is not None:
            missing = [fname for fname in list(query_fnames) + list(gallery_fnames) if fname not in fname_lookup]
            if missing:
                raise ValueError('LLCM metadata references files not present in identities: {}'.format(missing[:5]))
            self.query = [fname_lookup[fname] for fname in query_fnames]
            self.gallery = [fname_lookup[fname] for fname in gallery_fnames]
        else:
            if 'query' not in self.split or 'gallery' not in self.split:
                raise ValueError('LLCM splits.json must contain query/gallery identities or meta.json must contain query_fnames/gallery_fnames')
            self.query = _pluck_query(identities, self.split['query'])
            self.gallery = _pluck_gallery(identities, self.split['gallery'])

        if verbose:
            print(self.__class__.__name__, 'dataset loaded')
            print('  subset   | # ids | # images')
            print('  ---------------------------')
            print('  train    | {:5d} | {:8d}'.format(self.num_train_ids, len(self.train)))
            print('  val      | {:5d} | {:8d}'.format(self.num_val_ids, len(self.val)))
            print('  trainval | {:5d} | {:8d}'.format(self.num_trainval_ids, len(self.trainval)))
            print('  query    | {:5d} | {:8d}'.format(len(set(pid for _, pid, _ in self.query)), len(self.query)))
            print('  gallery  | {:5d} | {:8d}'.format(len(set(pid for _, pid, _ in self.gallery)), len(self.gallery)))
