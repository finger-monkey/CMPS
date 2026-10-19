from __future__ import absolute_import
import warnings
from .regdb import Regdb
from .sysu import Sysu
from .llcm import LLCM

warnings.filterwarnings('ignore')
__factory = {
    'regdb_v2': Regdb,
    'sysu_v2': Sysu,
    'llcm_v2': LLCM,
}


def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    if name not in __factory:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __factory[name](root, *args, **kwargs)


def get_dataset(name, root, *args, **kwargs):
    warnings.warn('get_dataset is deprecated. Use create instead.')
    return create(name, root, *args, **kwargs)
