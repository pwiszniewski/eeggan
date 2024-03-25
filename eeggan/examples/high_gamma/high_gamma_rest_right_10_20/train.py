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


from eeggan.examples.high_gamma.high_gamma_rest_right_10_20.make_data import FS, N_PROGRESSIVE_STAGES, INPUT_LENGTH
from eeggan.examples.high_gamma.models.baseline import Baseline
from eeggan.examples.high_gamma.train import train
from eeggan.model.builder import ProgressiveModelBuilder
from eeggan.pytorch.utils.weights import weight_filler
from eeggan.training.progressive.handler import ProgressionHandler
from eeggan.training.trainer.gan_softplus import GanSoftplusTrainer
from utils import read_config
from eeggan.examples.high_gamma.make_data import load_dataset
from eeggan.data.datasets.MK_gen import MK_gen

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

n_epochs_per_stage = 2000
default_config = dict(
    n_chans=21,  # number of channels in data
    n_classes=1,  # number of classes in data
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
    n_time=INPUT_LENGTH,

    upsampling='area',
    downsampling='area',
    discfading='cubic',
    genfading='cubic',
)


# data:
#     fs: 256
#     target: 'ssvep_dcgan'
#     dataset:
#         name: data.datasets.MK_gen.MK_gen
#         args: {}
#         kwargs:
#             dataset_dir: /zfsauton2/home/pwisznie/Datasets/MK_gen_231229
#             subjects_selected: [0]
#             channels_selected: ['1']
#             targets_selected: [8] # 8 13
#             overlap: 0
#             file_suffix: high
#             percent_of_data: 10

data = {
    'fs': 256,
    'dataset': {
        'name': 'data.datasets.MK_gen.MK_gen',
        'args': {},
        'kwargs': {
            'dataset_dir': None,
            'subjects_selected': [0],
            'channels_selected': ['1'],
            'targets_selected': [8],
            'overlap': 0,
            'file_suffix': 'high',
            'percent_of_data': 10
        }
    }
}

default_model_builder = Baseline(default_config['n_stages'], default_config['n_latent'], default_config['n_time'],
                                 default_config['n_chans'], default_config['n_classes'], default_config['n_filters'],
                                 upsampling=default_config['upsampling'], downsampling=default_config['downsampling'],
                                 discfading=default_config['discfading'], genfading=default_config['genfading'])


def run(subj_ind: int, result_name: str, dataset_path: str, deep4_path: str, result_path: str,
        config: dict = default_config, model_builder: ProgressiveModelBuilder = default_model_builder):
    dataset_org = load_dataset(subj_ind, dataset_path)
    dataset = MK_gen(**data['dataset']['kwargs'])
    dataset.train_data = dataset_org.train_data
    dataset.train_data.X = dataset.dataset.tensors[0]
    # append zeros to the last dimension to 896
    dataset.train_data.X = torch.cat((dataset.train_data.X, torch.zeros(dataset.train_data.X.size(0), 1, 896 - dataset.train_data.X.size(2))), dim=2)
    # replicate the data to 21 channels
    dataset.train_data.X = dataset.train_data.X.repeat(1, 21, 1)
    dataset.train_data.y = dataset.dataset.tensors[1].float()
    y_onehot = torch.zeros(dataset.train_data.y.size(0), config['n_classes'])
    dataset.train_data.y_onehot = y_onehot.scatter_(1, dataset.train_data.y.long().unsqueeze(1), 1)
    n_examples = 160
    # take first n_examples examples
    dataset.train_data.X = dataset.train_data.X[:n_examples]
    dataset.train_data.y = dataset.train_data.y[:n_examples]
    dataset.train_data.y_onehot = dataset.train_data.y_onehot[:n_examples]



    # assert 1<0


    result_path_subj = os.path.join(result_path, result_name, str(subj_ind))
    os.makedirs(result_path_subj, exist_ok=True)

    joblib.dump(config, os.path.join(result_path_subj, 'config.dict'), compress=False)
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

    train(subj_ind, dataset, deep4_path, result_path_subj, progression_handler, trainer, config['n_batch'],
          config['lr_d'], config['lr_g'], config['betas'], config['n_epochs_per_stage'], config['n_epochs_metrics'],
          config['plot_every_epoch'], config['orig_fs'])


if __name__ == '__main__':
    config = read_config()
    data['dataset']['kwargs']['dataset_dir'] = config['PATHS']['MK_gen_231229_path']

    run(subj_ind=config['TRAINING']['subj_ind'],
        result_name=config['TRAINING']['result_name'],
        dataset_path=config['PATHS']['dataset_path'],
        deep4_path=config['PATHS']['deep4_path'],
        result_path=config['PATHS']['result_path'])