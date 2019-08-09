#import pybm3d
from skimage import filters, measure, restoration, morphology
import numpy as np
import torch as pt
import os
import sys
sys.path.extend(['/media/data/Documents/Promotion/Project_Helpers/'])
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
from utils.registrator import Registrator
from pt_models import Generator
import importlib
import importlib.util
import h5py
import imreg_dft as ird

import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')
from time import time


def show(image, **kwargs):
    fig = plt.figure(figsize=kwargs.get('figsize', (8, 8)))
    plt.imshow(image, **kwargs)
    plt.axis('off')
    plt.show()


def hist(image):
    fig = plt.figure(figsize=(8, 8))
    plt.hist(image.ravel(), 100, range=(0.1, 1))
    plt.show()


def bilateral(image):
    sigma = restoration.estimate_sigma(image)*3
    denoised = restoration.denoise_bilateral(image.astype(np.float), sigma_color=sigma, multichannel=False)
    return denoised


def wavelet(image):
    sigma = restoration.estimate_sigma(image)*1.5
    denoised = restoration.denoise_wavelet(image.astype(np.float), multichannel=False, sigma=sigma)
    return denoised


def nl_means(image):
    sigma = restoration.estimate_sigma(image)*1.5
    denoised = restoration.denoise_nl_means(image.astype(np.float), h=sigma, multichannel=False)
    return denoised


def median(image, filter_size=1):
    filter = morphology.disk(filter_size)
    image = filters.median(image.astype(np.float), selem=filter)
    image = image / image.max()

    return image


def bm3d(image):
    sigma = restoration.estimate_sigma(image)*2
    denoised = pybm3d.bm3d.bm3d(image, sigma)
    denoised[np.isinf(denoised)] = 0
    denoised[np.isnan(denoised)] = 0
    denoised[denoised < 0] = 0
    return denoised


def measurement_preparation(datapoint):
    image = datapoint.image if datapoint.mask is None else datapoint.image * datapoint.mask
    reference = datapoint.reference if datapoint.mask is None else datapoint.reference * datapoint.mask
    if datapoint.transformation is not None:
        image = ird.transform_img_dict(image, datapoint.transformation, bgval=0.0, order=3)

    return image, reference


def psnr(datapoint):
    image, reference = measurement_preparation(datapoint)
    return measure.compare_psnr(reference, image)


def ssim(datapoint):
    image, reference = measurement_preparation(datapoint)
    return measure.compare_ssim(image, reference)


def cnr(datapoint):

    rois, background = [datapoint.image[roi > 0] for roi in datapoint.rois], datapoint.image[datapoint.background > 0]
    background_mean = background.mean()
    background_std = background.std()
    cnrs = []
    for roi in rois:
        cnrs.append(np.abs(roi.mean() - background_mean) / np.sqrt(0.5*(roi.std()**2 + background_std)**2))
    cnrs = np.array(cnrs)

    return cnrs.mean()


def msr(datapoint):

    rois = [datapoint.image[roi > 0] for roi in datapoint.rois]
    msrs = []
    for roi in rois:
        mean = roi.mean()
        std = roi.std()
        msrs.append(mean/std)
    msrs = np.array(msrs)

    return msrs.mean()


class CycGAN(object):

    def __init__(self, checkpoint, config):
        dirname = os.path.dirname(config)
        sys.path.extend([dirname])
        config = os.path.basename(config).split('.')[0]
        spec = importlib.util.spec_from_file_location(config, os.path.join(dirname, config)+'.py')
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        self.denoiser = Generator(**config.generator)
        self.checkpoints = [os.path.join(dirname, ckpt) for ckpt in os.listdir(dirname) if '.pt' in ckpt]
        checkpoint = pt.load(self.checkpoints[checkpoint])['model']
        checkpoint = {'.'.join(key.split('.')[1:]): value for key, value in checkpoint.items()
                      if 'generator_hq' in key}
        self.denoiser.load_state_dict(checkpoint)

    def __call__(self, image):

        denoiser = self.denoiser
        image = pt.from_numpy(image.copy())[None, None, ...]
        with pt.no_grad():
            output = denoiser(image)[0, 0].numpy()
        return output


class Datapoint(object):

    def __init__(self, key=None, image=None, reference=None, method=None, background=None, rois=None, mask=None,
                 transformation=None):
        self.data = {}
        self.key = key
        self.method = method
        self.image = image.copy()
        self.mask = mask
        self.reference = reference
        self.rois = rois
        self.background = background
        self.contains_measurement = False
        self.transformation = transformation
        self.compute_time = 0

    @property
    def key(self):
        return self.data['key']

    @key.setter
    def key(self, key):
        self.data['key'] = key

    @property
    def method(self):
        return self.data['method']

    @method.setter
    def method(self, method):
        self.data['method'] = method

    @property
    def compute_time(self):
        return self.data['compute_time']

    @compute_time.setter
    def compute_time(self, duration):
        self.data['compute_time'] = duration

    def extract_information(self):

        # test if some data is missing
        assert self.contains_measurement, f'no measurement was performed on this datapoint!'

        return self.data

    def add_measurement(self, name, value):
        self.data[name] = value
        self.contains_measurement = True

    def copy(self):

        new_datapoint = Datapoint(image=self.image,
                                  reference=self.reference,
                                  rois=self.rois,
                                  background=self.background,
                                  key=self.key,
                                  method=self.method,
                                  mask=self.mask,
                                  transformation=self.transformation)
        new_datapoint.data = self.data.copy()

        return new_datapoint


class Analysis(object):

    def __init__(self, lq, hq, methods, measurements, output_path, n_processes=1,
                 preprocess=lambda x: x, export_denoised=None):

        self.lq = self.open_storage(lq)
        self.hq = self.open_storage(hq)
        self.n_processes = n_processes
        self.export_denoised = export_denoised
        self.output_path = output_path
        self.output_dir = os.path.dirname(output_path)
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        self.measurements = dict(measurements)
        self.methods = dict(methods)
        self.preprocess = preprocess
        self.registrator = Registrator()
        self.lq_keys = list(self.lq.keys())
        self.hq_keys = list(self.hq.keys())

        # late binding because of mp
        self.hq.close()
        self.lq.close()
        self.lq = lq
        self.hq = hq

    @staticmethod
    def open_storage(filename):
        return h5py.File(filename, 'r', swmr=True, libver='latest')

    @staticmethod
    def get_rois(image, registrator):

        masked = registrator.segment(image, offset=1)
        masked = np.pad(masked, ((6, 6),), 'constant', constant_values=0)
        contours = registrator.get_contours(masked, offset=0.4, min_length=0.1)
        masks = [registrator.get_mask(contour, masked.shape)[6:-6, 6:-6] for contour in contours]

        return masks

    @staticmethod
    def get_background(image, registrator):

        inverted_mask = np.ones_like(image)
        inverted_mask[registrator.segment(image, offset=0) > 0] = 0

        return inverted_mask

    def apply_measurements(self, datapoint):

        for measurement_name, measurement in self.measurements.items():
            result = measurement(datapoint)
            datapoint.add_measurement(measurement_name, result)

        result = datapoint.extract_information()
        return result

    def apply_denoising(self, method_name, datapoint):

        method = self.methods[method_name]
        datapoint = datapoint.copy()
        start = time()
        image = method(datapoint.image)
        duration = time() - start
        datapoint.image = image
        datapoint.compute_time = duration
        datapoint.method = method_name

        return datapoint

    def export_images(self, datapoints, index):

        indices = list(range(len(datapoints[1:])))
        index = str(index)
        exported = {'index': index,
                    'key': datapoints[0].key}
        filedir = os.path.join(self.export_denoised, index)
        if not os.path.isdir(filedir):
            os.makedirs(filedir)

        # save original and reference
        datapoint = datapoints[0]
        image = datapoint.image.copy()
        plt.figure(figsize=(5.12, 5.12), frameon=False)
        plt.imshow(image, 'gray', interpolation='none')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(filedir, 'original.png'), dpi=100)
        plt.close()
        plt.clf()
        plt.cla()
        image = datapoint.reference.copy()
        plt.figure(figsize=(5.12, 5.12), frameon=False)
        plt.imshow(image, 'gray', interpolation='none')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(filedir, 'reference.png'), dpi=100)
        plt.close()
        plt.clf()
        plt.cla()

        for datapoint in datapoints[1:]:
            inner_index = indices.pop(np.random.randint(len(indices)))
            filename = os.path.join(filedir, f'{inner_index}.png')

            image = datapoint.image.copy()
            plt.figure(figsize=(5.12, 5.12), frameon=False)
            plt.imshow(image, 'gray', interpolation='none')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(filename, dpi=100)
            exported[datapoint.method] = f'{inner_index}.png'
            plt.close()
            plt.clf()
            plt.cla()

        return exported

    def __call__(self, acceptance=None):

        lq = self.open_storage(self.lq)
        hq = self.open_storage(self.hq)

        print('beginning analysis...')

        total_slices = len(self.lq_keys)

        print(f'\tnumber of datapoins:\t{total_slices}')

        # initialize result list for accumulating measurements
        results = []
        exported = []
        indices = list(range(len(self.lq_keys)))

        # cycle over samples
        for i, key in enumerate(self.lq_keys[150:]):

            i+= 150
            # skip failed registrations
            entry = lq[key]
            if acceptance is not None:
                if entry.attrs['difference'] > acceptance or np.isnan(entry.attrs['difference']):
                    continue

            transform = dict(entry.attrs)
            transform.pop('difference')
            transform.pop('frames')
            image = self.preprocess(entry.value)
            reference = self.preprocess(hq[self.hq_keys[i]].value)

            rois = self.get_rois(reference, self.registrator)
            background = self.get_background(reference, self.registrator)
            mask = self.registrator.segment(reference, offset=-1)

            # initialize original image as datapoint
            datapoints = [Datapoint(key=key, reference=reference,
                                    rois=rois, background=background,
                                    method='original', image=image, transformation=transform, mask=mask)]

            # generate all denoised images
            pool = mp.Pool(self.n_processes)
            denoising = partial(self.apply_denoising, datapoint=datapoints[0])
            datapoints += pool.map(denoising, self.methods.keys())

            # perform measurements on all denoised images
            pool = mp.Pool(self.n_processes)
            results += pool.map(self.apply_measurements, datapoints)

            if self.export_denoised is not None:
                index = indices.pop(np.random.randint(len(indices)))
                exported_row = self.export_images(datapoints, index)
                exported.append(exported_row)

            print('\r\tprogress: \t{}%'.format(round(100 * i / total_slices, 2)), end='')

        # convert results to a dataframe and save
        results_frame = pd.DataFrame(data=results)
        results_frame.to_csv(self.output_path)
        if self.export_denoised is not None:
            exported_frame = pd.DataFrame(data=exported)
            exported_frame.to_csv(os.path.join(self.export_denoised, 'exports.csv'))

        hq.close()
        lq.close()

        return results_frame


if __name__ == '__main__':

    checkpoint = 132
    config = '/media/network/DL_PC/Results/ilja/pt-cycoct/run_024/config.py'

    methods = {'median': median,
               'ours': CycGAN(checkpoint, config),
               'wavelet': wavelet,
               'bilateral': bilateral,
               'nl_means': nl_means,
               'bm3d': bm3d}
    measurements = {'PSNR': psnr,
                    'CNR': cnr,
                    'MSR': msr,
                    'SSIM': ssim}
    savefile = './measurements.csv'
    lq = '/media/network/DL_PC/Datasets/oct_quality_validation/low.hdf5'
    hq = '/media/network/DL_PC/Datasets/oct_quality_validation/high.hdf5'


    def preprocess(image):
        image = image / image.max()
        mean = image[image>0].mean()
        std = image[image>0].std()
        level = mean - 0.5*std
        image = np.clip(image, level, 1.0) - level
        image = image / image.max()
        return image


    analysis = Analysis(lq, hq, methods, measurements, savefile, 4, preprocess=preprocess, export_denoised='./exports/')
    results = analysis()

