import os
import json
import importlib
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse

import timesead

from utils.utils import recursive_update


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_path', type=str, help='Path to the checkpoint file', required=True)
    parser.add_argument('-l', '--log_dir', help='Path to the logging directory. Defaults to the directory of the checkpoint')
    parser.add_argument('-t', '--plot_threshold', action='store_true', help='Plot threshold line for anomaly')
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    log_dir = os.path.dirname(checkpoint_path)
    if args.log_dir is not None:
        log_dir = args.log_dir
    config_path = os.path.join(log_dir, 'config.json')
    results_path = os.path.join(log_dir, 'results.json')

    with open(config_path, 'r') as ff:
        config = json.load(ff)
    with open(results_path, 'r') as ff:
        results = json.load(ff)

    data_import = importlib.import_module(f'datasets.{config["dataset"]}')
    experiment_import = importlib.import_module(f'experiments.{config["experiment"]}')

    train_transform = recursive_update(experiment_import.TRAIN_PIPELINE, config['transforms']['train'])
    test_transform = recursive_update(experiment_import.TEST_PIPELINE, config['transforms']['test'])

    print('Setting up data module')
    data_module = data_import.DATASET(config['data_params'], config['batch_size'], train_transform, test_transform)
    data_module.prepare_data()
    data_module.setup('test')

    print('Loading model from', checkpoint_path)
    loaded_data = torch.load(checkpoint_path)
    if 'model' not in loaded_data:
        model = experiment_import.MODEL(config['transforms']['seq_len'],
                                        data_module.num_features,
                                        config['model_params'],
                                        config['run_params'])
        model.detector = loaded_data['detector']
    else:
        model = torch.load(checkpoint_path)['model']

    print('Computing anomaly scores')
    labels, scores = model.detector.get_labels_and_scores(data_module.test_dataloader())
    labels = labels.tolist()
    scores = scores.tolist()

    plot_dir = os.path.join(log_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    print('Plotting anomalies')
    timesead.utils.plot_utils.plot_sequence_against_anomaly(scores, labels, scatter=False)
    if args.plot_threshold:
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
        if args.plot_threshold:
            plt.axhline(y=results['best_ts_f1_score']['other_info']['threshold'], color='dimgray', linestyle='--')
        plt.savefig(os.path.join(plot_dir, f'anomaly_{index}.png'))
        plt.close()
