from utils.utils import recursive_update, get_final_config, convert_str_to_objects
from utils.lightning_utils import SaveConfigCallback

import torch
import lightning.pytorch as lp
from lightning.pytorch.loggers import TensorBoardLogger

from typing import Optional, List
import os
import sys
import importlib
import argparse
import copy
import json


def _sanity_check_checkpoint(checkpoint_path: str, config: dict):
    checkpoint_config_path = os.path.join(os.path.dirname(checkpoint_path), 'config.json')
    if not os.path.exists(checkpoint_config_path):
        print('Checkpoint is not in logging directory, skipping config check')
        return

    config1, config2 = None, None
    def remove_non_training_data(config: dict):
        del config['data_params']['test_ids']
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
    parser.add_argument('--log_dir', type=str, help='The base directory to save logs in', required=False, default='lightning_logs')
    parser.add_argument('--device', type=str, help='Device to run on', required=False, default='auto')

    args = parser.parse_args()
    run_info = args.__dict__.copy()
    run_info['run_command'] = ' '.join(sys.argv)

    overrides = args.overrides or []
    if args.seed:
        overrides.append(f'seed={args.seed}')
    config = get_final_config(args.config, overrides)
    raw_config = copy.deepcopy(config)
    convert_str_to_objects(config)
    print('Running experiment', config['experiment'], 'on dataset', config['dataset'])
    print('Final config:', config)

    if not config['experiment'] or not config['dataset']:
        raise ValueError('Experiment and dataset must be specified in the config')


    lp.seed_everything(config['seed'])

    # Setting up the logger and loading the dataset and model
    testing_ids = ','.join([str(id) for id in config['data_params']['test_ids']])
    run_name = f'{config["experiment"]}_{config["dataset"]}_{testing_ids}'
    if args.suffix:
        run_name += f'_{args.suffix}'
    logger = TensorBoardLogger(args.log_dir, name=run_name)
    data_import = importlib.import_module(f'datasets.{config["dataset"]}')
    experiment_import = importlib.import_module(f'experiments.{config["experiment"]}')

    train_transform = recursive_update(experiment_import.TRAIN_PIPELINE, config['transforms']['train'])
    test_transform = recursive_update(experiment_import.TEST_PIPELINE, config['transforms']['test'])
    if config['transforms']['train'] or config['transforms']['test']:
        print('Train transform pipeline:', train_transform)
        print('Test transform pipeline:', test_transform)
    data_module = data_import.DATASET(config['data_params'], config['batch_size'], train_transform, test_transform)
    if args.checkpoint_path:
        _sanity_check_checkpoint(args.checkpoint_path, raw_config)
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
    callbacks = [SaveConfigCallback(raw_config, run_info)]
    for callback in config['run_params']['callbacks']:
        callbacks.append(callback['class'](**callback['args']))
    trainer = lp.Trainer(
        max_epochs=config['epochs'],
        logger=logger,
        deterministic=True,
        callbacks=callbacks,
        enable_progress_bar=not args.disable_progress_bar,
        accelerator=args.device
    )
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

