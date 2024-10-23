from typing import List, Optional
import collections
import copy
import yaml

# These imports are needed for dynamic class instantiation
import torch
import lightning.pytorch as lp


DEFAULT_CONFIG = {
    'seed': 10,
    'data_params': {
        # Fraction of the testing data to use for validation when there is no separate validation set
        'validation_split': 0.3,
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

def apply_config_updates(config: dict, updates: Optional[List[str]]):
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
        # If the value is not a string, evaluate it
        current[key] = value if value[0].isalpha() else eval(value)
    return config

def get_final_config(config_paths: List[str], updates: list):
    config = copy.deepcopy(DEFAULT_CONFIG)
    for config_path in config_paths:
        config = recursive_update(config, _parse_config(config_path))
    config = apply_config_updates(config, updates)
    return config


def convert_str_to_objects(config: dict):
    """Convert class: 'module.class' to class: module.class inside the dictionary."""
    for key, value in config.items():
        if isinstance(value, dict):
            convert_str_to_objects(value)
        elif isinstance(value, list):
            if isinstance(value[0], dict):
                for item in value:
                    convert_str_to_objects(item)
        elif isinstance(value, str):
            if key == 'class':
                config[key] = eval(value)
            # Convert lambda functions
            if value.startswith('lambda'):
                config[key] = eval(value)


def recursive_update(dict1, dict2):
    """
    Given two dictionaries dict1 and dict2, update dict1 recursively.
    """
    for key, value in dict2.items():
        if key == 'args':
            dict1[key] = dict2[key]
        elif isinstance(value, collections.abc.Mapping):
            result = recursive_update(dict1.get(key, {}), value)
            dict1[key] = result
        else:
            dict1[key] = dict2[key]
    return dict1
