from utils.utils import recursive_update
from utils.lightning_utils import SaveConfigCallback

import torch
import lightning.pytorch as lp
from lightning.pytorch.loggers import TensorBoardLogger

from typing import Optional, List
import os
import sys
import importlib
import argparse
import yaml
import copy
import json

DEFAULT_CONFIG = {
    'seed': 10,
    'data_params': {
        'data_splits': [0.75, 0.25],
    },
    'transforms': {
        'train': {},
        'test': {},
        'seq_len': 0  # Should be updated by the experiment
    },
    'epochs': 1,
    'batch_size': 128,
    'model_params': {},
    'detector_params': {},
    'run_params': {
        'optimizer': {
            'class': 'torch.optim.Adam',
            'args': {
                'lr': 1e-3,
            },
        },
        'scheduler': {
            'class': 'torch.optim.lr_scheduler.StepLR',
            'args': {
                'step_size': 100,
                'gamma': 1,
            },
        },
        'callbacks': [
            {
                'class': 'lp.callbacks.EarlyStopping',
                'args': {'monitor': 'val_loss', 'patience': 10}
            }
        ]
    },
}

def _parse_config(config_path: str) -> dict:
    with open(config_path, 'r') as ff:
        return yaml.safe_load(ff)

def _apply_config_updates(config: dict, updates: Optional[List[str]]):
    if updates is None:
        return config
    for update in updates:
        key, value = update.split('=')
        current = config
        while '.' in key:
            first, key = key.split('.', 1)
            current = current[first]
        if key not in current:
            print(f'Creating new config entry with {update}')
        current[key] = eval(value)
    return config

def _convert_str_to_objects(config: dict):
    # Convert class: 'module.class' to class: module.class
    for key, value in config.items():
        if isinstance(value, dict):
            _convert_str_to_objects(value)
        elif isinstance(value, list):
            if isinstance(value[0], dict):
                for item in value:
                    _convert_str_to_objects(item)
        elif isinstance(value, str):
            if key == 'class':
                config[key] = eval(value)
            # Convert lambda functions
            if value.startswith('lambda'):
                config[key] = eval(value)

def _sanity_check_checkpoint(checkpoint_path: str, config: dict):
    checkpoint_config_path = os.path.join(os.path.dirname(checkpoint_path), 'config.json')
    if not os.path.exists(checkpoint_config_path):
        print('Checkpoint is not in logging directory, skipping config check')
        return

    config1, config2 = None, None
    def remove_non_training_data(config: dict):
        del config['data_params']['test_server_ids']
    with open(checkpoint_config_path, 'r') as ff:
        checkpoint_config = json.load(ff)
        config1 = copy.deepcopy(config)
        config2 = copy.deepcopy(checkpoint_config)
    remove_non_training_data(config1)
    remove_non_training_data(config2)
    if config1 != config2:
        print('Run config:', config1)
        print('Checkpoint config:', config2)
        raise ValueError('Configurations do not match between the checkpoint and the current run')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', help='Path to config files', required=True)
    parser.add_argument('--checkpoint_path', help='Path to the checkpoint file', required=False)
    parser.add_argument('--seed', type=int, help='Random seed', required=False)
    parser.add_argument('--test_only', action='store_true', help='Only run the test step')
    parser.add_argument('-o', '--overrides', nargs='*', help='Manual config updates in the form key1=value1')
    parser.add_argument('--disable_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--suffix', type=str, help='Add suffix to the run name', required=False)
    args = parser.parse_args()
    run_info = {'run_command': ' '.join(sys.argv)}

    # Configuration setup
    config = DEFAULT_CONFIG
    for config_path in args.config:
        config = recursive_update(config, _parse_config(config_path))
    if args.seed:
        config['seed'] = args.seed
    config = _apply_config_updates(config, args.overrides)
    config_dump = copy.deepcopy(config)
    _convert_str_to_objects(config)
    print('Final config:', config)

    if not config['experiment'] or not config['dataset']:
        raise ValueError('Experiment and dataset must be specified in the config')


    lp.seed_everything(config['seed'])

    # Setting up the logger and loading the dataset and model
    testing_server_ids = ','.join([str(id) for id in config['data_params']['test_server_ids']])
    run_name = f'{config["experiment"]}_{config["dataset"]}_{testing_server_ids}'
    if args.suffix:
        run_name += f'_{args.suffix}'
    logger = TensorBoardLogger('lightning_logs',
                               name=run_name)
    data_import = importlib.import_module(f'datasets.{config["dataset"]}')
    experiment_import = importlib.import_module(f'experiments.{config["experiment"]}')

    train_transform = recursive_update(experiment_import.TRAIN_PIPELINE, config['transforms']['train'])
    test_transform = recursive_update(experiment_import.TEST_PIPELINE, config['transforms']['test'])
    if config['transforms']['train'] or config['transforms']['test']:
        print('Train transform pipeline:', train_transform)
        print('Test transform pipeline:', test_transform)
    data_module = data_import.DATASET(config['data_params'], config['batch_size'], train_transform, test_transform)
    if args.checkpoint_path:
        _sanity_check_checkpoint(args.checkpoint_path, config_dump)
        loaded_data = torch.load(args.checkpoint_path)
        if 'model' not in loaded_data:
            model = experiment_import.MODEL(config['transforms']['seq_len'],
                                            data_module.num_features,
                                            config['model_params'],
                                            config['run_params'])
            model.detector = loaded_data['detector']
        else:
            model = torch.load(args.checkpoint_path)['model']
    else:
        model = experiment_import.MODEL(config['transforms']['seq_len'],
                                        data_module.num_features,
                                        config['model_params'],
                                        config['run_params'])


    # Training and testing
    callbacks = [SaveConfigCallback(config_dump, run_info)]
    for callback in config['run_params']['callbacks']:
        callbacks.append(callback['class'](**callback['args']))
    trainer = lp.Trainer(max_epochs=config['epochs'], logger=logger, deterministic=True,
                         callbacks=callbacks,
                         enable_progress_bar=not args.disable_progress_bar)
    if not args.test_only:
        trainer.fit(model=model, datamodule=data_module)

        model.setup_detector(config['detector_params'], data_module.val_dataloader())
        try:
            torch.save(dict(detector=model.detector), os.path.join(logger.log_dir, 'final_model.pth'))
        except Exception as ee:
            print('Exception occurred when saving model, trying to save just the detector')
            torch.save(dict(model=model, detector=model.detector), os.path.join(logger.log_dir, 'final_model.pth'))

    # Reseed for testing
    lp.seed_everything(config['seed'])
    trainer.test(model, datamodule=data_module)

