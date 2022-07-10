# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.
# Change dataloader multiprocess start method to anything not fork
import torch.multiprocessing as mp
try:
    mp.set_start_method('forkserver')  # Reuse process created
except RuntimeError:
    pass

import os
import sys
import json
import logging
from easydict import EasyDict as edict

import random
import numpy as np

# Torch packages
import torch

# Train deps
from config import get_config
import shutil

from lib.test import test
from lib.train import train
from lib.multitrain import train as train_mp
from lib.check_data import check_data
from lib.utils import load_state_with_same_shape, get_torch_device, count_parameters
from lib.dataset import initialize_data_loader, _init_fn
from lib.datasets import load_dataset
from lib.datasets.semantic_kitti import SemanticKITTI
from lib.datasets.Indoor3DSemSegLoader import S3DIS
from lib.datasets.nuscenes import Nuscenes
from lib.dataloader import InfSampler
import lib.transforms as t

from models import load_model

import MinkowskiEngine as ME    # force loadding

def setup_seed(seed):
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)
                random.seed(seed)
                torch.backends.cudnn.deterministic = True

def main():
    config = get_config()
    ch = logging.StreamHandler(sys.stdout)
    logging.getLogger().setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(config.log_dir, './model.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logging.basicConfig(
                format=os.uname()[1].split('.')[0] + ' %(asctime)s %(message)s',
                datefmt='%m/%d %H:%M:%S',
                handlers=[ch, file_handler])

    if config.test_config:
        # When using the test_config, reload and overwrite it, so should keep some configs 
        val_bs = config.val_batch_size
        is_export = config.is_export

        json_config = json.load(open(config.test_config, 'r'))
        json_config['is_train'] = False
        json_config['weights'] = config.weights
        json_config['multiprocess'] = False
        json_config['log_dir'] = config.log_dir
        json_config['val_threads'] = config.val_threads
        json_config['submit'] = config.submit
        config = edict(json_config)

        config.val_batch_size = val_bs
        config.is_export = is_export
        config.is_train = False
        sys.path.append(config.log_dir)
        # from local_models import load_model
    else:
        '''bakup files'''
        if not os.path.exists(os.path.join(config.log_dir,'models')):
            os.mkdir(os.path.join(config.log_dir,'models'))
        for filename in os.listdir('./models'):
                if ".py" in filename: # donnot cp the init file since it will raise import error
                    shutil.copy(os.path.join("./models", filename), os.path.join(config.log_dir,'models'))
                elif 'modules' in filename:
                    # copy the moduls folder also
                    if os.path.exists(os.path.join(config.log_dir,'models/modules')):
                        shutil.rmtree(os.path.join(config.log_dir,'models/modules'))
                    shutil.copytree(os.path.join('./models',filename), os.path.join(config.log_dir,'models/modules'))

        shutil.copy('./main.py', config.log_dir)
        shutil.copy('./config.py', config.log_dir)
        shutil.copy('./lib/train.py', config.log_dir)
        shutil.copy('./lib/test.py', config.log_dir)

    if config.resume == 'True':
        new_iter_size = config.max_iter
        new_bs = config.batch_size
        config.resume = config.log_dir
        json_config = json.load(open(config.resume + '/config.json', 'r'))
        json_config['resume'] = config.resume
        config = edict(json_config)
        config.weights = os.path.join(config.log_dir, 'weights.pth')   # use the pre-trained weights
        logging.info('==== resuming from {}, Total {} ======'.format(config.max_iter, new_iter_size))
        config.max_iter = new_iter_size
        config.batch_size = new_bs
    else:
        config.resume = None

    if config.is_cuda and not torch.cuda.is_available():
        raise Exception("No GPU found")
    gpu_list = range(config.num_gpu)
    device = get_torch_device(config.is_cuda)

    # torch.set_num_threads(config.threads)
    # torch.manual_seed(config.seed)
    # if config.is_cuda:
    #       torch.cuda.manual_seed(config.seed)

    logging.info('===> Configurations')
    dconfig = vars(config)
    for k in dconfig:
        logging.info('      {}: {}'.format(k, dconfig[k]))

    DatasetClass = load_dataset(config.dataset)
    logging.info('===> Initializing dataloader')

    setup_seed(2021)

    """
    ---- Setting up train, val, test dataloaders ----
    Supported datasets:
    - ScannetSparseVoxelizationDataset
    - ScannetDataset
    - SemanticKITTI
    """

    if config.is_train:
        if config.dataset == 'ScannetSparseVoxelizationDataset':
            train_data_loader = initialize_data_loader(
                    DatasetClass,
                    config,
                    phase=config.train_phase,
                    threads=config.threads,
                    augment_data=True,
                    elastic_distortion=config.train_elastic_distortion,
                    shuffle=True,
                    repeat=True,
                    batch_size=config.batch_size,
                    limit_numpoints=config.train_limit_numpoints)

            val_data_loader = initialize_data_loader(
                    DatasetClass,
                    config,
                    threads=config.val_threads,
                    phase=config.val_phase,
                    augment_data=False,
                    elastic_distortion=config.test_elastic_distortion,
                    shuffle=False,
                    repeat=False,
                    batch_size=config.val_batch_size,
                    limit_numpoints=False)
        elif config.dataset == "SemanticKITTI":
            dataset = SemanticKITTI(root=config.semantic_kitti_path,
                                   num_points = None,
                                   voxel_size=config.voxel_size,
                                   sample_stride=config.sample_stride,
                                   submit=False)
            collate_fn_factory = t.cfl_collate_fn_factory
            train_data_loader = torch.utils.data.DataLoader(
                dataset['train'],
                batch_size=config.batch_size,
                sampler=InfSampler(dataset['train'], shuffle=True), # shuffle=true, repeat=true
                num_workers=config.threads,
                pin_memory=True,
                collate_fn=collate_fn_factory(config.train_limit_numpoints)
            )

            val_data_loader = torch.utils.data.DataLoader( # shuffle=false, repeat=false
                dataset['test'], 
                batch_size=config.batch_size,
                num_workers=config.val_threads,
                pin_memory=True,
                collate_fn=t.cfl_collate_fn_factory(False) 
            )
        elif config.dataset == "S3DIS":
            trainset = S3DIS(
                    config,
                    train=True,
                    )
            valset = S3DIS(
                    config,
                    train=False,
                    )
            train_data_loader = torch.utils.data.DataLoader(
                trainset,
                batch_size=config.batch_size,
                sampler=InfSampler(trainset, shuffle=True), # shuffle=true, repeat=true
                num_workers=config.threads,
                pin_memory=True,
                collate_fn=t.cfl_collate_fn_factory(config.train_limit_numpoints)
            )

            val_data_loader = torch.utils.data.DataLoader( # shuffle=false, repeat=false
                valset,
                batch_size=config.batch_size,
                num_workers=config.val_threads,
                pin_memory=True,
                collate_fn=t.cfl_collate_fn_factory(False)
            )
        elif config.dataset == 'Nuscenes':
            config.xyz_input=False
            trainset = Nuscenes(
                    config,
                    train=True,
                    )
            valset = Nuscenes(
                    config,
                    train=False,
                    )
            train_data_loader = torch.utils.data.DataLoader(
                trainset,
                batch_size=config.batch_size,
                sampler=InfSampler(trainset, shuffle=True), # shuffle=true, repeat=true
                num_workers=config.threads,
                pin_memory=True,
                collate_fn=t.cfl_collate_fn_factory(False)
            )

            val_data_loader = torch.utils.data.DataLoader( # shuffle=false, repeat=false
                valset,
                batch_size=config.batch_size,
                num_workers=config.val_threads,
                pin_memory=True,
                collate_fn=t.cfl_collate_fn_factory(False)
            )
        else:
            print('Dataset {} not supported').format(config.dataset)
            raise NotImplementedError

        # Setting up num_in_channel and num_labels
        if train_data_loader.dataset.NUM_IN_CHANNEL is not None:
            num_in_channel = train_data_loader.dataset.NUM_IN_CHANNEL
        else:
            num_in_channel = 3

        num_labels = train_data_loader.dataset.NUM_LABELS

    else: # not config.is_train
        val_DatasetClass = load_dataset('ScannetDatasetWholeScene_evaluation')
        if config.dataset == 'ScannetSparseVoxelizationDataset':

            if config.is_export: # when export, we need to export the train results too
                train_data_loader = initialize_data_loader(
                    DatasetClass,
                    config,
                    phase=config.train_phase,
                    threads=config.threads,
                    augment_data=True,
                    elastic_distortion=config.train_elastic_distortion,  # DEBUG: not sure about this
                    shuffle=False,
                    repeat=False,
                    batch_size=config.batch_size,
                    limit_numpoints=config.train_limit_numpoints)

            val_data_loader = initialize_data_loader(
                    DatasetClass,
                    config,
                    threads=config.val_threads,
                    phase=config.val_phase,
                    augment_data=False,
                    elastic_distortion=config.test_elastic_distortion,
                    shuffle=False,
                    repeat=False,
                    batch_size=config.val_batch_size,
                    limit_numpoints=False)

            if val_data_loader.dataset.NUM_IN_CHANNEL is not None:
                num_in_channel = val_data_loader.dataset.NUM_IN_CHANNEL
            else:
                num_in_channel = 3

            num_labels = val_data_loader.dataset.NUM_LABELS
        elif config.dataset == "SemanticKITTI":
            dataset = SemanticKITTI(root=config.semantic_kitti_path,
                                   num_points = None,
                                   voxel_size=config.voxel_size,
                                   submit=config.submit)
            val_data_loader = torch.utils.data.DataLoader( # shuffle=false, repeat=false
                dataset['test'],
                batch_size=config.val_batch_size,
                num_workers=config.val_threads,
                pin_memory=True,
                collate_fn=t.cfl_collate_fn_factory(False)
            )
            num_in_channel = 4
            num_labels = 19

        elif config.dataset == 'S3DIS':
            config.xyz_input = False

            trainset = S3DIS(
                    config,
                    train=True,
                    )
            valset = S3DIS(
                    config,
                    train=False,
                    )
            train_data_loader = torch.utils.data.DataLoader(
                trainset,
                batch_size=config.batch_size,
                sampler=InfSampler(trainset, shuffle=True), # shuffle=true, repeat=true
                num_workers=config.threads,
                pin_memory=True,
                collate_fn=t.cfl_collate_fn_factory(config.train_limit_numpoints)
            )

            val_data_loader = torch.utils.data.DataLoader( # shuffle=false, repeat=false
                valset,
                batch_size=config.batch_size,
                num_workers=config.val_threads,
                pin_memory=True,
                collate_fn=t.cfl_collate_fn_factory(False)
            )
            num_in_channel = 9
            num_labels = 13
        elif config.dataset == 'Nuscenes':
            config.xyz_input = False
            trainset = Nuscenes(
                    config,
                    train=True,
                    )
            valset = Nuscenes(
                    config,
                    train-False,
                    )
            train_data_loader = torch.utils.data.DataLoader(
                trainset,
                batch_size=config.batch_size,
                sampler=InfSampler(trainset, shuffle=True), # shuffle=true, repeat=true
                num_workers=config.threads,
                pin_memory=True,
                collate_fn=t.cfl_collate_fn_factory(False)
            )

            val_data_loader = torch.utils.data.DataLoader( # shuffle=false, repeat=false
                valset,
                batch_size=config.batch_size,
                num_workers=config.val_threads,
                pin_memory=True,
                collate_fn=t.cfl_collate_fn_factory(False)
            )
            num_in_channel = 5
            num_labels = 16
        else:
            print('Dataset {} not supported').format(config.dataset)
            raise NotImplementedError

    logging.info('===> Building model')

    NetClass = load_model(config.model)
    model = NetClass(num_in_channel, num_labels, config)
    logging.info('===> Number of trainable parameters: {}: {}M'.format(NetClass.__name__,count_parameters(model)/1e6))
    logging.info(model)

    # Set the number of threads
    # ME.initialize_nthreads(12, D=3)

    model = model.to(device)

    if config.weights == 'modelzoo':    # Load modelzoo weights if possible.
        logging.info('===> Loading modelzoo weights')
        model.preload_modelzoo()

    # Load weights if specified by the parameter.
    elif config.weights.lower() != 'none':
        logging.info('===> Loading weights: ' + config.weights)
        state = torch.load(config.weights)
        # delete the keys containing the 'attn' since it raises size mismatch
        d_ = {k:v for k,v in state['state_dict'].items() if '_map' not in k } # debug: sometiems model conmtains 'map_qk' which is not right for naming a module, since 'map' are always buffers
        d = {}
        for k in d_.keys():
            if 'module.' in k:
                d[k.replace('module.','')] = d_[k]
            else:
                d[k] = d_[k]
        # del d_

        if config.weights_for_inner_model:
            model.model.load_state_dict(d)
        else:
            if config.lenient_weight_loading:
                matched_weights = load_state_with_same_shape(model, state['state_dict'])
                model_dict = model.state_dict()
                model_dict.update(matched_weights)
                model.load_state_dict(model_dict)
            else:
                model.load_state_dict(d, strict=True)

    if config.is_debug:
        check_data(model, train_data_loader, val_data_loader, config)
        return None
    elif config.is_train:
        if config.multiprocess:
            train_mp(NetClass, train_data_loader, val_data_loader, config)
        else:
            train(model, train_data_loader, val_data_loader, config)
    elif config.is_export:
        test(model, train_data_loader, config, save_pred=True, split='train')
        test(model, val_data_loader, config, save_pred=True, split='val')
    else:
        assert config.multiprocess == False
        # if test for submission, make a submit directory at current directory
        submit_dir = os.path.join(os.getcwd(), 'submit', 'sequences')
        if config.submit and not os.path.exists(submit_dir):
            os.makedirs(submit_dir)
            print("Made submission directory: " + submit_dir)
        test(model, val_data_loader, config, submit_dir=submit_dir)

if __name__ == '__main__':
    __spec__ = None
    main()
