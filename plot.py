import os
import glob
import json
import importlib
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse

import timesead

from utils.utils import recursive_update

def _update_config_if_needed(config: dict):
    data_params = config['data_params']
    if 'train_server_ids' not in data_params:
        return
    new_data_params = {}
    for key, val in data_params.items():
        if 'server_ids' in key:
            new_data_params[key.replace('server_ids', 'ids')] = val
        else:
            new_data_params[key] = val
    new_data_params['num_workers'] = 4
    config['data_params'] = new_data_params

def generate_plots_for_log(log_dir: str, plot_threshold: bool):
    model_path = os.path.join(log_dir, 'final_model.pth')
    if not os.path.exists(model_path):
        with open(os.path.join(log_dir, 'run.json'), 'r') as ff:
            model_path = json.load(ff)['checkpoint_path']

    config_path = os.path.join(log_dir, 'config.json')
    results_path = os.path.join(log_dir, 'results.json')
    with open(config_path, 'r') as ff:
        config = json.load(ff)
    with open(results_path, 'r') as ff:
        results = json.load(ff)

    _update_config_if_needed(config)

    data_import = importlib.import_module(f'datasets.{config["dataset"]}')
    experiment_import = importlib.import_module(f'experiments.{config["experiment"]}')

    train_transform = recursive_update(experiment_import.TRAIN_PIPELINE, config['transforms']['train'])
    test_transform = recursive_update(experiment_import.TEST_PIPELINE, config['transforms']['test'])
    batch_dim = experiment_import.BATCH_DIM

    print('Setting up data module')
    data_module = data_import.DATASET(
        config['data_params'],
        config['batch_size'],
        train_transform,
        test_transform,
        batch_dim=batch_dim,
    )
    data_module.prepare_data()
    data_module.setup('test')

    print('Loading model from', model_path)
    loaded_data = torch.load(model_path)
    if 'model' not in loaded_data:
        model = experiment_import.MODEL(config['transforms']['seq_len'],
                                        data_module.num_features,
                                        config['model_params'],
                                        config['run_params'])
        model.detector = loaded_data['detector']
    else:
        model = torch.load(model_path)['model']

    print('Computing anomaly scores')
    labels, scores = model.detector.get_labels_and_scores(data_module.test_dataloader())
    labels = labels.tolist()
    scores = scores.tolist()

    plot_dir = os.path.join(log_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    print('Plotting anomalies')
    timesead.utils.plot_utils.plot_sequence_against_anomaly(scores, labels, scatter=False)
    if plot_threshold:
        plt.axhline(y=results['best_ts_f1_score']['other_info']['threshold'], color='dimgray', linestyle='--')
    plt.savefig(os.path.join(plot_dir, f'full.png'))
    plt.close()

    boundaries = 2 * np.array(labels + [0]) - np.array([0] + labels)
    lbs = np.argwhere(boundaries > 1).flatten()
    rbs = np.argwhere(boundaries < 0).flatten()

    for index, (lb, rb) in enumerate(zip(lbs, rbs)):
        len_window = rb - lb
        left = max(0, lb-len_window//2)
        right = min(len(scores), rb+len_window//2)
        timesead.utils.plot_utils.plot_sequence_against_anomaly(
            scores[left:right],
            labels[left:right],
            scatter=False,
            xticks=[0, len_window//2, 3*len_window//2, 2*len_window]
            )
        if plot_threshold:
            plt.axhline(y=results['best_ts_f1_score']['other_info']['threshold'], color='dimgray', linestyle='--')
        plt.savefig(os.path.join(plot_dir, f'anomaly_{index}.png'))
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log_dirs', help='Glob path to log directories', required=True)
    parser.add_argument('-t', '--plot_threshold', action='store_true', help='Plot threshold line for anomaly')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing plots')
    args = parser.parse_args()

    for log_dir in glob.glob(args.log_dirs):
        if os.path.exists(os.path.join(log_dir, 'plots')) and not args.overwrite:
            print('Plots already exist for', log_dir)
            continue
        print('Generating plots for', log_dir)
        try:
            generate_plots_for_log(log_dir, args.plot_threshold)
        except Exception as e:
            print('Error generating plots for', log_dir, e)
            continue
