"""
This script is used to run hyperparameter tuning using Optuna.
It takes in config files and overrides as input. \
The search parameters are defined in the tuning module for the experiment (tuning.{experiment}).
"""
import argparse
import sys
import importlib
import copy
import os
import json
from typing import List

import torch
import lightning.pytorch as lp
import optuna
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor

from utils.utils import get_final_config, recursive_update, convert_str_to_objects, apply_config_updates
from utils.lightning_utils import SaveConfigCallback


def objective(trial: optuna.trial.Trial, tuning_modules: List[str], base_config: dict, run_info: dict):
    trial_overrides = []
    for module in tuning_modules:
        tuning_import = importlib.import_module(f'tuning.{module}')
        trial_generate_fn = tuning_import.generate_trial_overrides
        trial_overrides += trial_generate_fn(trial)

    base_config = copy.deepcopy(base_config)
    config = apply_config_updates(base_config, trial_overrides)
    raw_config = copy.deepcopy(config)
    convert_str_to_objects(config)


    if not config['experiment'] or not config['dataset']:
        raise ValueError('Experiment and dataset must be specified in the config')

    print('Running experiment', config['experiment'], 'on dataset', config['dataset'])
    print('Final config:', config)


    lp.seed_everything(config['seed'])

    # Setting up the logger and loading the dataset and model
    run_name = f'{config["experiment"]}_{config["dataset"]}'
    logger = TensorBoardLogger('optuna_logs', name=run_name)
    data_import = importlib.import_module(f'datasets.{config["dataset"]}')
    experiment_import = importlib.import_module(f'experiments.{config["experiment"]}')

    train_transform = recursive_update(experiment_import.TRAIN_PIPELINE, config['transforms']['train'])
    test_transform = recursive_update(experiment_import.TEST_PIPELINE, config['transforms']['test'])
    data_module = data_import.DATASET(config['data_params'], config['batch_size'], train_transform, test_transform)
    model = experiment_import.MODEL(config['transforms']['seq_len'],
                                    data_module.num_features,
                                    config['model_params'],
                                    config['run_params'],
                                    plot_anomalies=False)


    # Training and testing
    callbacks = [SaveConfigCallback(raw_config, run_info), LearningRateMonitor(logging_interval='epoch')]
    for callback in config['run_params']['callbacks']:
        callbacks.append(callback['class'](**callback['args']))
    trainer = lp.Trainer(
        max_epochs=config['epochs'],
        logger=logger,
        deterministic=True,
        callbacks=callbacks,
        enable_progress_bar=not run_info['disable_progress_bar'],
        accelerator=run_info['device']
    )

    trainer.fit(model=model, datamodule=data_module)
    model.setup_detector(config['detector_params'], data_module.val_dataloader())

    trainer.test(model, datamodule=data_module)

    with open(os.path.join(logger.log_dir, 'results.json'), 'r') as ff:
        results = json.load(ff)

    return results[run_info['metric']]['score']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--configs', nargs='+', help='Path to defualt config files', required=True)
    parser.add_argument('-o', '--overrides', nargs='*', help='Manual config updates in the form key1=value1')
    parser.add_argument('-t', '--tune', nargs='*', help='Which modules to tune on. By defualt, parameter search is done on the experiment module (tuning.<experiment>)')
    parser.add_argument('--disable_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--resume', action='store_true', help='Resume existing study')
    parser.add_argument('--device', type=str, help='Device to run on', required=False, default='auto')
    parser.add_argument('--metric', type=str, help='Metric to optimize', default='best_ts_f1_score')
    parser.add_argument('--num_trials', type=int, help='Number of trials to run', default=50)
    args = parser.parse_args()
    run_info = args.__dict__.copy()
    run_info['optuna_run'] = True
    run_info['run_command'] = ' '.join(sys.argv)

    base_config = get_final_config(args.configs, args.overrides)
    tuning_modules = args.tune or [base_config['experiment']]
    objective_fn = lambda trial: objective(trial, tuning_modules, base_config, run_info)


    study_name = '_'.join(tuning_modules) + f'_{base_config["dataset"]}'
    db_path = 'sqlite:///optuna.db'
    if not args.resume:
        try:
            optuna.delete_study(study_name=study_name, storage=db_path)
        except KeyError:
            pass
    study = optuna.create_study(
        study_name=study_name,
        storage=db_path,
        load_if_exists=True
        )
    study.optimize(objective_fn, n_trials=args.num_trials)
    print(study.best_params)
