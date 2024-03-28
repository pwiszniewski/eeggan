import os
from abc import ABCMeta
from typing import List, Tuple, TypeVar, Generic, Dict

import numpy as np
import scipy
import torch
from ignite.metrics import Metric
from torch import Tensor
from torch.nn.modules.module import Module

from eeggan.cuda import to_device
from eeggan.data.preprocess.resample import upsample
from eeggan.training.trainer.trainer import BatchOutput, Trainer
from eeggan.validation.metrics.frechet import calculate_activation_statistics, calculate_frechet_distances
from eeggan.validation.metrics.inception import calculate_inception_score
from eeggan.validation.metrics.wasserstein import create_wasserstein_transform_matrix, \
    calculate_sliced_wasserstein_distance
from eeggan.validation.validation_helper import logsoftmax_act_to_softmax, compute_spectral_amplitude

"""
save ouputs of the model
"""
T = TypeVar('T')


class SaveOuputsMat(metaclass=ABCMeta):
    def __init__(self, out_path: str, prefix: str, fs: float):
        self.path = out_path
        self.prefix = prefix
        self.fs = fs

    def __call__(self, trainer: Trainer):
        batch_output: BatchOutput = trainer.state.output
        X_real = batch_output.batch_real.X.data.cpu().numpy()
        X_fake = batch_output.batch_fake.X.data.cpu().numpy()

        n_samples = X_real.shape[2]
        freqs = np.fft.rfftfreq(n_samples, 1. / self.fs)
        amps_real = compute_spectral_amplitude(X_real, axis=2)
        amps_real_mean = amps_real.mean(axis=(0, 1)).squeeze()
        amps_real_std = amps_real.std(axis=(0, 1)).squeeze()
        amps_fake = compute_spectral_amplitude(X_fake, axis=2)
        amps_fake_mean = amps_fake.mean(axis=(0, 1)).squeeze()
        amps_fake_std = amps_fake.std(axis=(0, 1)).squeeze()

        data = {'real': X_real, 'fake': X_fake, 'freqs': freqs, 'amps_real_mean': amps_real_mean,
                'amps_real_std': amps_real_std, 'amps_fake_mean': amps_fake_mean, 'amps_fake_std': amps_fake_std}
        mat_path = os.path.join(self.path, self.prefix + str(trainer.state.epoch) + '.mat')
        scipy.io.savemat(mat_path, data)




