import os

import h5py
import torch as pt
from torch.utils.data import Dataset


class OCTQualityDatasetHDF5(Dataset):

    def __init__(self, hn, ln, storage_dir, vmin=0.0, vmax=1.0, whitening=False, preprocess=True, transforms=None,
                 mode='classification'):

        self._preprocess = preprocess
        self.storage_dir = storage_dir
        self.hn = hn
        self.ln = ln
        hn = h5py.File(os.path.join(self.storage_dir, hn), 'r')
        ln = h5py.File(os.path.join(self.storage_dir, ln), 'r')
        self.keys = {'hn': list(hn.keys()), 'ln': list(ln.keys())}
        hn.close()
        ln.close()  # hdf5 files are closed and reopened later, otherwise there are problems with dataloader workers
        self.vmin = vmin
        self.vmax = vmax
        self.whitening = whitening
        self.transforms = transforms
        self.mode = mode

    def init(self):
        """
        delayed init due to problems with hdf5 and multiprocessing in dataloader that arise otherwise
        :return: None
        """

        self.hn = h5py.File(os.path.join(self.storage_dir, self.hn), 'r')
        self.ln = h5py.File(os.path.join(self.storage_dir, self.ln), 'r')

    def __len__(self):

        return min(len(self.keys['lq']), len(self.keys['hq']))

    def __getitem__(self, item):

        # late init because of how hdf5 and dataloader work together
        if isinstance(self.hn, str) or isinstance(self.ln, str):
            self.init()

        lq_key = self.keys['lq'][item]
        lq = [self.preprocess(self.hn[lq_key].value),
              self.hn[lq_key].attrs['frames'] * pt.ones(1)]
        hq_key = self.keys['hq'][item]
        hq = [self.preprocess(self.ln[hq_key].value),
              self.ln[hq_key].attrs['frames'] * pt.ones(1)]

        if self.mode == 'classification':
            # convert number of frames to classes for classification
            hq[1] = hq[1] / 12
            hq[1][hq[1] > 1] = 2

            lq[1] = lq[1] / 12
            lq[1][lq[1] > 1] = 2

        sample = ((lq[0], hq[0]), (lq[1].int(), hq[1].int()))
        return sample

    def preprocess(self, image):
        if self._preprocess:
            image = image / image.max()
            vmin = self.parse_threshold(self.vmin, image)
            vmax = self.parse_threshold(self.vmax, image)
            image = image.clip(vmin, vmax)
            image -= vmin
            image = image / image.max()

            if self.transforms:
                image = self.transforms(image)

            if self.whitening:
                image = image - image.mean()
                image = image / image.std()

        return pt.from_numpy(image[None, ...])

    def close(self):

        if self.hn is not None: self.hn.close()
        if self.ln is not None: self.ln.close()

    @staticmethod
    def parse_threshold(threshold, image):

        if isinstance(threshold, float):
            return threshold
        elif threshold == 'mean':
            return image[image > 0].mean()
        elif isinstance(threshold, str) and 'mean' in threshold:
            factor = float(threshold[4:])
            std = image[image > 0].std()
            mean = image[image > 0].mean()
            return mean + factor * std