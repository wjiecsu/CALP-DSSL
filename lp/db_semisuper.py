# Authors: A. Iscen, G. Tolias, Y. Avrithis, O. Chum. 2018.

import torch.utils.data as data

from PIL import Image

import os
import os.path
import sys
import pdb

import numpy as np
import time

import faiss
from faiss import normalize_L2

from .diffusion import *
import scipy
import torch.nn.functional as F
import torch

import scipy.stats

import pickle



def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    #按类别寻找，
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images


class DatasetFolder(data.Dataset):
    #继承dataSet类
    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        classes, class_to_idx = self._find_classes(root)
        #classes表示字母类型，idx表示为0，1，2
        samples = make_dataset(root, class_to_idx, extensions)
        #samples[0]表示路径,samples[1]表示 编号1，2，3
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))
        self.root = root
        self.loader = loader
        self.extensions = extensions
        self.classes = classes
        self.num_class = len(classes)
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.transform = transform
        self.target_transform = target_transform
        self.images = None

    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        if (index not in self.labeled_idx):
            target = self.p_labels[index]
        weight = self.p_weights[index]

        sample = self.loader(path)


        if self.transform is not None:
            sample = self.transform(sample)
        c_weight = self.class_weights[target]
        return sample, target, weight, c_weight

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def PIL_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
def default_loader(path):
    return PIL_loader(path)


class DBSS(DatasetFolder):
    #继承数据类+自己的属性
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        super(DBSS, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform = transform,
                                          target_transform = target_transform)
        self.imgs = self.samples
        self.labeled_idx = []
        self.unlabeled_idx = []
        self.all_labels = []
        self.p_labels = []
        self.p_weights = np.ones((len(self.imgs),))
        self.class_weights = np.ones((len(self.classes),),dtype = np.float32)