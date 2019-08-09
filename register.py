import h5py
import numpy as np

from utils.registrator import Registrator


def progress(i, total):

    fraction = i / total
    bar_length = 50
    bar_fraction = int(fraction*bar_length)
    bar = '#'*bar_fraction + '='*(bar_length-bar_fraction)
    print(f'\r|{bar}| {round(fraction*100, 2)}%\t', end='')


def preprocess(sample, factor=0.5):
    sample = sample/sample.max()
    mean = sample[sample>0].mean()
    std = sample[sample>0].std()
    level = mean - factor*std
    sample = sample.clip(level, 1.0) - level
    sample = sample/sample.max()
    sample[0:8] = 0
    return sample


storage_dir = '/media/network/DL_PC/Datasets/oct_quality_validation/'
lq = h5py.File(storage_dir+'low.hdf5', 'r')
hq = h5py.File(storage_dir+'high.hdf5', 'r')
output = h5py.File(storage_dir+'registered.hdf5', 'a')

registrator = Registrator(order=3, offset=0)

lq_keys, hq_keys = list(lq.keys()), list(hq.keys())
n_samples = len(lq_keys)

try:
    for i, (lq_key, hq_key) in enumerate(zip(lq_keys, hq_keys)):

        subject, template = preprocess(lq[lq_key].value), preprocess(hq[hq_key].value)
        try:
            registered, transformation, difference = registrator.register_single(template, subject)
            transformation.pop('timg')
        except (IndexError, ValueError):
            print(f'\nsomething went wrong in the registration of {lq_key}')
            registered = np.zeros_like(subject)
            difference = 1
            transformation = {}
        output[lq_key] = registered
        for key, value in transformation.items():
            output[lq_key].attrs[key] = value
        output[lq_key].attrs['difference'] = difference
        # output.flush()
        progress(i, n_samples)
finally:
    output.close()
    lq.close()
    hq.close()

"""
something went wrong in the registration of 17182-624913-00
something went wrong in the registration of 17182-624913-01
something went wrong in the registration of 17182-624913-02
something went wrong in the registration of 17802-626829-14
something went wrong in the registration of 17802-626829-15
something went wrong in the registration of 7681-624748-00
something went wrong in the registration of 7681-624748-01
something went wrong in the registration of 7681-624748-10
something went wrong in the registration of 7681-624748-11
something went wrong in the registration of 7681-624748-13
"""