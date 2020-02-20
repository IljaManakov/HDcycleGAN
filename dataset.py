"""
Copyright 2019 Ilja Manakov

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
from os.path import join

import h5py
import torch as pt
from torch.utils.data import Dataset
from imageio import imread


class OCTQualityDatasetHDF5(Dataset):

    def __init__(self, hn, ln, storage_dir, vmin=0.0, vmax=1.0, whitening=False, preprocess=True, transforms=None):

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

    def init(self):
        """
        delayed init due to problems with hdf5 and multiprocessing in dataloader that arise otherwise
        :return: None
        """

        self.hn = h5py.File(os.path.join(self.storage_dir, self.hn), 'r')
        self.ln = h5py.File(os.path.join(self.storage_dir, self.ln), 'r')

    def __len__(self):

        return min(len(self.keys['ln']), len(self.keys['hn']))

    def __getitem__(self, item):

        # late init because of how hdf5 and dataloader work together
        if isinstance(self.hn, str) or isinstance(self.ln, str):
            self.init()

        hn_key = self.keys['hn'][item]
        hn = [self.preprocess(self.hn[hn_key][()]),
              self.hn[hn_key].attrs['frames'] * pt.ones(1)]
        ln_key = self.keys['ln'][item]
        ln = [self.preprocess(self.ln[ln_key][()]),
              self.ln[ln_key].attrs['frames'] * pt.ones(1)]

        # convert number of frames to classes for classification
        ln[1] = ln[1] / 12
        ln[1][ln[1] > 1] = 2

        hn[1] = hn[1] / 12
        hn[1][hn[1] > 1] = 2

        sample = ((hn[0], ln[0]), (hn[1].long(), ln[1].long()))
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


class OCTQualityDataset(Dataset):

    def __init__(self, parent_folder, fraction=0.8, transformation=lambda x: x):

        self.transformation = transformation

        # get lists of filenames
        self.hn_files = self.gather_filenames(join(parent_folder, 'high-noise'))
        self.ln_files = self.gather_filenames(join(parent_folder, 'low-noise'))

        # keep fraction
        last_index = int(len(self.hn_files) * abs(fraction))
        self.hn_files = self.hn_files[:last_index] if fraction > 0 else self.hn_files[-last_index:]
        last_index = int(len(self.ln_files) * abs(fraction))
        self.ln_files = self.ln_files[:last_index] if fraction > 0 else self.ln_files[-last_index:]

    @staticmethod
    def gather_filenames(folder):

        # walk through directories and collect filenames with path (in numerical order to preserve pairing)
        filenames = []
        for root, dirs, files in os.walk(folder):
            dirs.sort(key=int)
            if not files: continue
            files.sort(key=lambda x: int(x.split('.')[0]))
            filenames += [f'{root}/{f}' for f in files]
        return filenames

    def prepare_image(self, filename):

        # load and convert to normalized float32 tensor
        image = imread(filename)
        image = self.transformation(image)
        image = pt.from_numpy(image)[None, ...].float()
        image = image / image.max()
        return image

    def __len__(self):
        return min(len(self.hn_files), len(self.ln_files))

    def __getitem__(self, item):

        hn = self.prepare_image(self.hn_files[item])
        ln = self.prepare_image(self.ln_files[item])
        images = (hn, ln)
        labels = (1, 2)

        return images, labels
