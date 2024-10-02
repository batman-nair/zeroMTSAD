import argparse
import sys
import importlib
import copy

import torch
import lightning.pytorch as lp
import optuna
from lightning.pytorch.loggers import TensorBoardLogger

from utils.utils import get_final_config, recursive_update, convert_str_to_objects
from utils.lightning_utils import SaveConfigCallback


def objective(trial: optuna.trial.Trial, config_files, run_info):
    hidden_dimensions = trial.suggest_categorical(
        'hidden_dimensions',
        [
            "[30]",
            "[40]",
            "[30, 30]",
            "[50, 50]"
        ])
    window_size = trial.suggest_int('window_size', 5, 100)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)

    overrides = [
        f'model_params.hidden_dimensions={hidden_dimensions}',
        f'transforms.train.window.args.window_size={window_size}',
        f'transforms.test.window.args.window_size={window_size}',
        f'transforms.seq_len={window_size}',
        f'run_params.optimizer.args.lr={learning_rate}',
    ]

    config = get_final_config(config_files, overrides)
    raw_config = copy.deepcopy(config)
    convert_str_to_objects(config)


    if not config['experiment'] or not config['dataset']:
        raise ValueError('Experiment and dataset must be specified in the config')

    print('Running experiment', config['experiment'], 'on dataset', config['dataset'])
    print('Final config:', config)

    lp.seed_everything(config['seed'])

    # Setting up the logger and loading the dataset and model
    testing_server_ids = ','.join([str(id) for id in config['data_params']['test_server_ids']])
    run_name = f'{config["experiment"]}_{config["dataset"]}_{testing_server_ids}'
    logger = TensorBoardLogger('lightning_logs', name=run_name)
    data_import = importlib.import_module(f'datasets.{config["dataset"]}')
    experiment_import = importlib.import_module(f'experiments.{config["experiment"]}')

    train_transform = recursive_update(experiment_import.TRAIN_PIPELINE, config['transforms']['train'])
    test_transform = recursive_update(experiment_import.TEST_PIPELINE, config['transforms']['test'])
    data_module = data_import.DATASET(config['data_params'], config['batch_size'], train_transform, test_transform)
    model = experiment_import.MODEL(config['transforms']['seq_len'],
                                    data_module.num_features,
                                    config['model_params'],
                                    config['run_params'])


    # Training and testing
    callbacks = [SaveConfigCallback(raw_config, run_info)]
    for callback in config['run_params']['callbacks']:
        callbacks.append(callback['class'](**callback['args']))
    trainer = lp.Trainer(max_epochs=config['epochs'], logger=logger, deterministic=True,
                         callbacks=callbacks,
                         enable_progress_bar=False)

    trainer.fit(model=model, datamodule=data_module)

    return trainer.callback_metrics['val_loss'].item()


if __name__ == '__main__':
    run_info = {
        'optuna_run': True,
        'run_command': ' '.join(sys.argv),
    }

    config_files = ['configs/smd/smd.yml', 'configs/smd/lstm_ae.yml']
    objective_fn = lambda trial: objective(trial, config_files, run_info)

    study = optuna.create_study()
    study.optimize(objective_fn, n_trials=10)
    print(study.best_params)
