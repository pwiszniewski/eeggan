#  Author: Kay Hartmann <kg.hartma@gmail.com>
import os

import joblib
import torch
from ignite.engine import Events

import os
import sys
current_path = os.getcwd()
print(current_path)
sys.path.append("../eeggan")

import time

from eeggan.examples.high_gamma.high_gamma_rest_right_10_20.make_data import FS, N_PROGRESSIVE_STAGES, INPUT_LENGTH
from eeggan.examples.high_gamma.models.baseline import Baseline
from eeggan.examples.high_gamma.train import train
from eeggan.model.builder import ProgressiveModelBuilder
from eeggan.pytorch.utils.weights import weight_filler
from eeggan.training.progressive.handler import ProgressionHandler
from eeggan.training.trainer.gan_softplus import GanSoftplusTrainer
from utils import read_config, get_experiment_prefix
from eeggan.examples.high_gamma.make_data import load_dataset
from eeggan.data.datasets.MK_gen import MK_gen
from eeggan.data.datasets.SSVEP_12JFPM import SSVEP_12JFPM
from eeggan.data.datasets.SSVEP_our_lab_sessions import SSVEP_our_lab_sessions

import wandb

# n_epochs_per_stage = 2000
# default_config = dict(
#     n_chans=21,  # number of channels in data
#     n_classes=2,  # number of classes in data
#     orig_fs=FS,  # sampling rate of data
#
#     n_batch=128,  # batch size
#     n_stages=N_PROGRESSIVE_STAGES,  # number of progressive stages
#     n_epochs_per_stage=n_epochs_per_stage,  # epochs in each progressive stage
#     n_epochs_metrics=100,
#     plot_every_epoch=100,
#     n_epochs_fade=int(0.1 * n_epochs_per_stage),
#     use_fade=False,
#     freeze_stages=True,
#
#     n_latent=200,  # latent vector size
#     r1_gamma=10.,
#     r2_gamma=0.,
#     lr_d=0.005,  # discriminator learning rate
#     lr_g=0.001,  # generator learning rate
#     betas=(0., 0.99),  # optimizer betas
#
#     n_filters=120,
#     n_time=INPUT_LENGTH,
#
#     upsampling='area',
#     downsampling='area',
#     discfading='cubic',
#     genfading='cubic',
# )

################################# MK gen data #############################################################




n_epochs_per_stage = 2000 #2000
default_config = dict(
    n_chans=1, # 1 21,  # number of channels in data
    n_classes=None, # 1(MK_gen) 2(original) 12(12JFPM)  # number of classes in data, selected randomly
    orig_fs=FS,  # sampling rate of data

    n_batch=128,  # batch size
    n_stages=N_PROGRESSIVE_STAGES,  # number of progressive stages
    n_epochs_per_stage=n_epochs_per_stage,  # epochs in each progressive stage
    n_epochs_metrics=100,
    plot_every_epoch=100,
    n_epochs_fade=int(0.1 * n_epochs_per_stage),
    use_fade=False,
    freeze_stages=True,

    n_latent=200,  # latent vector size
    r1_gamma=10.,
    r2_gamma=0.,
    lr_d=0.005,  # discriminator learning rate
    lr_g=0.001,  # generator learning rate
    betas=(0., 0.99),  # optimizer betas

    n_filters=120,
    n_time=256, # INPUT_LENGTH(896), 256

    upsampling='area',
    downsampling='area',
    discfading='cubic',
    genfading='cubic',
)


def run(subj_ind: int, dataset_path: str, deep4_path: str, config: dict = default_config):
        # model_builder: ProgressiveModelBuilder = default_model_builder):
    n_examples = 1000
    # n_examples = 'all'
    plot_y_lim = None # (-3, 1)

    dataset_org = load_dataset(subj_ind, dataset_path)

    ################## original data ############################
    # dataset = dataset_org

    ################## MK gen data ############################
    # dataset_config = {
    #     'fs': 256,
    #     'dataset': {
    #         'name': 'data.datasets.MK_gen.MK_gen',
    #         'args': {},
    #         'kwargs': {
    #             'dataset_dir': 'P:\Datasets\MK_gen_231229',
    #             'subjects_selected': [0],
    #             'channels_selected': ['1'],
    #             'targets_selected': [8],  # 8 13
    #             'overlap': 0,
    #             'file_suffix': 'high',
    #             'percent_of_data': 10
    #         }
    #     }
    # }
    # dataset = MK_gen(**dataset_config['dataset']['kwargs'])
    # config['orig_fs'] = dataset.fs
    # config['n_classes'] = 1

    # ################# 12 JFPM SSVEP data ############################
    # dataset_config = {
    #     'dataset': {
    #         'name': 'data.datasets.SSVEP_12JFPM.SSVEP_12JFPM',
    #         'args': {},
    #         'kwargs': {
    #             'dataset_dir': 'H:\\AI\\Datasets\\12JFPM_SSVEP\\data',
    #             'subjects_selected': [8],
    #             'channels_selected': ['Oz'],
    #             'targets_selected': [9.25],
    #             'overlap': 255,
    #             # 'percent_of_best_snr': 10
    #         }
    #     }
    # }
    # dataset = SSVEP_12JFPM(**dataset_config['dataset']['kwargs'])
    # config['orig_fs'] = dataset.fs
    # config['n_classes'] = 12

    ################# our lab sessions data ############################
    dataset_config = {
        'dataset': {
            'name': 'data.datasets.SSVEP_our_lab_sessions.SSVEP_our_lab_sessions',
            'args': {},
            'kwargs': {
                'dataset_dir': 'H:\\AI\\Datasets\\Sesje_SSVEP\\240319_PW_6_7_8Hz',
                'file_prefixes': 'session_19_03_SSVEP_',
                'channels_selected': ['Oz'],
                # 'channels_selected': ['O1', 'Oz', 'O2', 'Cz', 'Fp1', 'ObokOka', 'Kark', 'Policzek', 'Szczeka'],
                'targets_selected': [7],
                'overlap': 0,
            }
        }
    }
    dataset = SSVEP_our_lab_sessions(**dataset_config['dataset']['kwargs'])
    # change targets to 0, dataset is TensorDataset
    dataset.dataset = torch.utils.data.TensorDataset(dataset.dataset.tensors[0],
                                                     torch.zeros_like(dataset.dataset.tensors[1]))

    config['orig_fs'] = dataset.fs
    config['n_classes'] = 1

    ############## prepare dataset ##############################
    dataset.train_data = dataset_org.train_data
    dataset.train_data.X = dataset.dataset.tensors[0]
    # # append zeros to the last dimension to 896
    # dataset.train_data.X = torch.cat((dataset.train_data.X, torch.zeros(dataset.train_data.X.size(0), 1, 896 - dataset.train_data.X.size(2))), dim=2)
    # repeat samples to config['n_time']
    # dataset.train_data.X = dataset.train_data.X.repeat(1, 1, config['n_time'] // dataset.train_data.X.size(2) + 1)
    # dataset.train_data.X = dataset.train_data.X[:, :, :config['n_time']]
    # # replicate the data to 21 channels
    # dataset.train_data.X = dataset.train_data.X.repeat(1, 21, 1)
    dataset.train_data.y = dataset.dataset.tensors[1].float()
    y_onehot = torch.zeros(dataset.train_data.y.size(0), config['n_classes'])
    dataset.train_data.y_onehot = y_onehot.scatter_(1, dataset.train_data.y.long().unsqueeze(1), 1)
    # take first n_examples examples
    if n_examples == 'all':
        n_examples = dataset.train_data.X.size(0)
    dataset.train_data.X = dataset.train_data.X[:n_examples]
    dataset.train_data.y = dataset.train_data.y[:n_examples]
    dataset.train_data.y_onehot = dataset.train_data.y_onehot[:n_examples]

    ################# data normalization ############################
    dataset.train_data.X = (dataset.train_data.X - dataset.train_data.X.mean()) / dataset.train_data.X.std()

    ################# add noise to data ############################
    # noise = torch.randn_like(dataset.train_data.X) * 0.1 * dataset.train_data.X.std()
    # dataset.train_data.X += noise

    ##############################################################

    print(f'X shape: {dataset.train_data.X.shape}')

    # save config
    with open(os.path.join(result_path_subj, 'config.dict'), 'w') as f:
        for key, value in config.items():
            f.write(f'{key}: {value}\n')

    # create model builder
    model_builder = Baseline(default_config['n_stages'], default_config['n_latent'],
                             default_config['n_time'],
                             default_config['n_chans'], default_config['n_classes'],
                             default_config['n_filters'],
                             upsampling=default_config['upsampling'],
                             downsampling=default_config['downsampling'],
                             discfading=default_config['discfading'], genfading=default_config['genfading'])

    joblib.dump(model_builder, os.path.join(result_path_subj, 'model_builder.jblb'), compress=True)

    # create discriminator and generator modules
    discriminator = model_builder.build_discriminator()
    generator = model_builder.build_generator()

    # initiate weights
    generator.apply(weight_filler)
    discriminator.apply(weight_filler)

    # trainer engine
    trainer = GanSoftplusTrainer(10, discriminator, generator, config['r1_gamma'], config['r2_gamma'])

    # handles potential progression after each epoch
    progression_handler = ProgressionHandler(discriminator, generator, config['n_stages'], config['use_fade'],
                                             config['n_epochs_fade'], freeze_stages=config['freeze_stages'])
    progression_handler.set_progression(0, 1.)
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), progression_handler.advance_alpha)

    generator.train()
    discriminator.train()

    # train(subj_ind, dataset, deep4_path, result_path_subj, progression_handler, trainer, config['n_batch'],
    #       config['lr_d'], config['lr_g'], config['betas'], config['n_epochs_per_stage'], config['n_epochs_metrics'],
    #       config['plot_every_epoch'], config['orig_fs']
    train(subj_ind=subj_ind, dataset=dataset, deep4s_path=deep4_path, result_path=result_path_subj,
          progression_handler=progression_handler, trainer=trainer, n_batch=config['n_batch'], lr_d=config['lr_d'],
          lr_g=config['lr_g'], betas=config['betas'], n_epochs_per_stage=config['n_epochs_per_stage'],
          n_epochs_metrics=config['n_epochs_metrics'], plot_every_epoch=config['plot_every_epoch'],
          plot_y_lim=plot_y_lim, orig_fs=config['orig_fs'], n_epochs_save_output=n_epochs_per_stage,
          plot_spectral_log_scale=False)

def format_time_seconds(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))


if __name__ == '__main__':
    config = read_config()
    experiment_name = get_experiment_prefix() + '_' + config['TRAINING']['result_name']
    result_path_subj = os.path.join(config['PATHS']['result_path'], experiment_name, config['TRAINING']['subj_ind'])
    os.makedirs(result_path_subj, exist_ok=True)

    # add empty file for notes
    with open(os.path.join(result_path_subj, 'info.txt'), 'w') as f:
        pass

    # Initialize wandb
    run_wandb = wandb.init(project="eeggan", config=config, name=experiment_name)

    # save output file
    with open(os.path.join(result_path_subj, 'out.txt'), 'w') as f:
        f.write(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        project_wandb = wandb.run.get_url()
        f.write(f"Project wandb: {project_wandb}\n")

    start_time = time.time()
    print(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    run(subj_ind=config['TRAINING']['subj_ind'],
        dataset_path=config['PATHS']['dataset_path'],
        deep4_path=config['PATHS']['deep4_path'])
    end_time = time.time()
    print(f"Training ended at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training took {format_time_seconds(end_time - start_time)}")

    # save output file
    with open(os.path.join(result_path_subj, 'out.txt'), 'a') as f:
        f.write(f"Training ended at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training took {format_time_seconds(end_time - start_time)}\n")

    run_wandb.finish()