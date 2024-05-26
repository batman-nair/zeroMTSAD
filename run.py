from utils.utils import recursive_update
from utils.lightning_utils import SaveConfigCallback

import lightning.pytorch as lp
from lightning.pytorch.loggers import TensorBoardLogger

from typing import Optional, List
import importlib
import argparse
import yaml

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
        current[key] = eval(value)
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', help='Path to config files', required=True)
    parser.add_argument('--checkpoint_path', help='Path to the checkpoint file', required=False)
    parser.add_argument('--seed', type=int, help='Random seed', required=False)
    parser.add_argument('--test_only', action='store_true', help='Only run the test step')
    parser.add_argument('-o', '--overrides', nargs='*', help='Manual config updates in the form key1=value1')
    parser.add_argument('--disable_progress_bar', action='store_true', help='Disable progress bar')
    args = parser.parse_args()

    # Configuration setup
    config = DEFAULT_CONFIG
    for config_path in args.config:
        config = recursive_update(config, _parse_config(config_path))
    if args.seed:
        config['seed'] = args.seed
    config = _apply_config_updates(config, args.overrides)

    if not config['experiment'] or not config['dataset']:
        raise ValueError('Experiment and dataset must be specified in the config')

    # Setting up the logger and loading the dataset and model
    testing_server_ids = ','.join([str(id) for id in config['data_params']['test_server_ids']])
    logger = TensorBoardLogger('lightning_logs',
                               name=f'{config["experiment"]}_{config["dataset"]}_{testing_server_ids}')
    data_import = importlib.import_module(f'datasets.{config["dataset"]}')
    experiment_import = importlib.import_module(f'experiments.{config["experiment"]}')

    train_transform = recursive_update(experiment_import.TRAIN_PIPELINE, config['transforms']['train'])
    test_transform = recursive_update(experiment_import.TEST_PIPELINE, config['transforms']['test'])
    data_module = data_import.DATASET(config['data_params'], train_transform, test_transform)
    if args.checkpoint_path:
        model = experiment_import.MODEL.load_from_checkpoint(args.checkpoint_path, seq_len=config['transforms']['seq_len'],
                                                             num_features=data_module.num_features, model_params=config['model_params'])
    else:
        model = experiment_import.MODEL(config['transforms']['seq_len'], data_module.num_features, config['model_params'])

    # Training and testing
    trainer = lp.Trainer(max_epochs=config['epochs'], logger=logger, deterministic=True,
                         callbacks=[SaveConfigCallback(config),
                                    lp.callbacks.EarlyStopping(monitor='val_loss')],
                         enable_progress_bar=not args.disable_progress_bar)
    if not args.test_only:
        trainer.fit(model=model, datamodule=data_module)

    trainer.test(model, datamodule=data_module)

