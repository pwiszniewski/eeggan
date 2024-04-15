#  Author: Kay Hartmann <kg.hartma@gmail.com>

import os
from typing import Tuple

import numpy as np
import scipy
import torch
from braindecode.torch_ext.modules import IntermediateOutputWrapper
from ignite.engine import Events
from ignite.metrics import MetricUsage
from matplotlib import pyplot
from torch import Tensor, optim
from torch.utils.data import DataLoader

from eeggan.cuda import to_cuda, init_cuda
from eeggan.data.dataset import Data
from eeggan.data.preprocess.resample import downsample
from eeggan.examples.high_gamma.make_data import load_deeps4
from eeggan.training.handlers.metrics import WassersteinMetric, InceptionMetric, FrechetMetric, LossMetric, \
    ClassificationMetric
from eeggan.training.handlers.plots import SpectralPlot
from eeggan.training.handlers.outputs import SaveOutputsMat
from eeggan.training.progressive.handler import ProgressionHandler
from eeggan.training.trainer.trainer import Trainer
from eeggan.data.dataset import Dataset
from eeggan.validation.validation_helper import compute_spectral_amplitude


def train(subj_ind: int, dataset: Dataset, deep4s_path: str, result_path: str,
          progression_handler: ProgressionHandler, trainer: Trainer, n_batch: int, lr_d: float, lr_g: float,
          betas: Tuple[float, float], n_epochs_per_stage: int, n_epochs_metrics: int, plot_every_epoch: int,
          plot_y_lim: Tuple[float, float], orig_fs: float, n_epochs_save_output: int, plot_spectral_log_scale: bool):
    plot_path = os.path.join(result_path, "plots")
    os.makedirs(plot_path, exist_ok=True)

    init_cuda()  # activate cuda

    train_data = dataset.train_data
    # test_data = dataset.test_data

    discriminator = progression_handler.discriminator
    generator = progression_handler.generator
    discriminator, generator = to_cuda(discriminator, generator)

    # usage to update every epoch and compute once at end of stage
    usage_metrics = MetricUsage(Events.STARTED, Events.EPOCH_COMPLETED(every=n_epochs_per_stage),
                                Events.EPOCH_COMPLETED(every=n_epochs_metrics))

    ##################### save real data and spectrum #####################
    X_real = train_data.X.numpy()
    n_samples = X_real.shape[2]
    fs = orig_fs
    freqs = np.fft.rfftfreq(n_samples, 1. / fs)
    spectral_amps_real = compute_spectral_amplitude(X_real, axis=2, log_scale=False)
    # save spectrum
    data = {'X_real': X_real, 'freqs': freqs, 'spectral_real': spectral_amps_real}
    mat_path = os.path.join(result_path, 'real_spectral.mat')
    scipy.io.savemat(mat_path, data)
    ##################### save real data and spectrum #####################

    for stage in range(progression_handler.n_stages):
        # optimizer
        optim_discriminator = optim.Adam(progression_handler.get_trainable_discriminator_parameters(), lr=lr_d,
                                         betas=betas)
        optim_generator = optim.Adam(progression_handler.get_trainable_generator_parameters(), lr=lr_g, betas=betas)
        trainer.set_optimizers(optim_discriminator, optim_generator)

        # modules to save
        to_save = {'discriminator': discriminator, 'generator': generator,
                   'optim_discriminator': optim_discriminator, 'optim_generator': optim_generator}

        # load trained deep4s for stage
        deep4s = load_deeps4(subj_ind, stage, deep4s_path)
        select_modules = ['conv_4', 'softmax']
        deep4s = [to_cuda(IntermediateOutputWrapper(select_modules, deep4)) for deep4 in deep4s]

        # scale data for current stage
        sample_factor = 2 ** (progression_handler.n_stages - stage - 1)
        X_block = downsample(train_data.X, factor=sample_factor, axis=2)

        # initiate spectral plotter
        spectral_plot = SpectralPlot(pyplot.figure(), plot_path, "spectral_stage_%d_" % stage, X_block.shape[2],
                                     orig_fs / sample_factor, y_lim=plot_y_lim, log_scale=plot_spectral_log_scale)
        event_name = Events.EPOCH_COMPLETED(every=plot_every_epoch)
        spectral_handler = trainer.add_event_handler(event_name, spectral_plot)

        # initiate output saver
        save_output = SaveOutputsMat(result_path, "output_stage_%d_" % stage, fs=orig_fs / sample_factor)
        # trainer.add_event_handler(Events.EPOCH_COMPLETED(every=n_epochs_save_output), save_output)

        # initiate metrics
        metric_wasserstein = WassersteinMetric(100, np.prod(X_block.shape[1:]).item())
        metric_inception = InceptionMetric(deep4s, sample_factor)
        metric_frechet = FrechetMetric(deep4s, sample_factor)
        metric_loss = LossMetric()
        metric_classification = ClassificationMetric(deep4s, sample_factor)
        metrics = [metric_wasserstein] # [metric_wasserstein, metric_inception, metric_frechet, metric_loss, metric_classification]
        metric_names = ["wasserstein"] # ["wasserstein", "inception", "frechet", "loss", "classification"]

        trainer.attach_metrics(metrics, metric_names, usage_metrics)

        # wrap into cuda loader
        train_data_tensor: Data[Tensor] = Data(
            *to_cuda(Tensor(X_block), Tensor(train_data.y), Tensor(train_data.y_onehot)))
        train_loader = DataLoader(train_data_tensor, batch_size=n_batch, shuffle=True)

        # train stage
        state = trainer.run(data=train_loader,
                            max_epochs=(stage + 1) * n_epochs_per_stage)
        trainer.remove_event_handler(spectral_plot, event_name)  # spectral_handler.remove() does not work :(

        ############################ save fake data and spectrum ############################
        # create fake data
        n_fake_examples = 1000
        with torch.no_grad():
            latent, y_fake, y_onehot_fake = generator.create_latent_input(rng=np.random.RandomState(0),
                                                                          n_trials=n_fake_examples)
            # X_fake = generator(latent, y=y_fake, y_onehot=y_onehot_fake) Tensor for argument #2 'mat1' is on CPU, but expected it to be on GPU (while checking arguments for addmm)
            X_fake = generator(latent.cuda(), y=y_fake.cuda(), y_onehot=y_onehot_fake.cuda())
        X_fake = X_fake.cpu().numpy()
        # calculate spectrum
        n_samples = X_fake.shape[2]
        fs = orig_fs / sample_factor
        freqs = np.fft.rfftfreq(n_samples, 1. / fs)
        spectral_amps_fake = compute_spectral_amplitude(X_fake, axis=2, log_scale=False)
        # save spectrum
        data = {'X_fake': X_fake, 'freqs': freqs, 'spectral_fake': spectral_amps_fake}
        mat_path = os.path.join(result_path, 'fake_spectral_stage_%d.mat' % stage)
        scipy.io.savemat(mat_path, data)
        ############################ save fake data and spectrum ############################

        # save stuff
        torch.save(to_save, os.path.join(result_path, 'modules_stage_%d.pt' % stage))
        torch.save(dict([(name, to_save[name].state_dict()) for name in to_save.keys()]),
                   os.path.join(result_path, 'states_stage_%d.pt' % stage))
        torch.save(trainer.state.metrics, os.path.join(result_path, 'metrics_stage_%d.pt' % stage))

        # advance stage if not last
        trainer.detach_metrics(metrics, usage_metrics)
        if stage != progression_handler.n_stages - 1:
            progression_handler.advance_stage()

