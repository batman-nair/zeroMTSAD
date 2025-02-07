from typing import List

import optuna


def generate_trial_overrides(trial: optuna.trial.Trial) -> List[str]:
    n_trees = trial.suggest_int('n_trees', 100, 500, step=100)
    sample_size = trial.suggest_int('sample_size', 128, 512, step=128)
    extension_level = trial.suggest_int('extension_level', -1, 10, step=1)
    window_size = trial.suggest_int('window_size', 5, 20, step=5)
    step_size = trial.suggest_int('step_size', 1, 5, step=1)

    overrides = [
        f'model_params.n_trees={n_trees}',
        f'model_params.sample_size={sample_size}',
        f'model_params.extension_level={extension_level}',
        f'transforms.train.window.args.window_size={window_size}',
        f'transforms.test.window.args.window_size={window_size}',
        f'transforms.seq_len={window_size}',
        f'transforms.train.window.args.step_size={step_size}',
        f'transforms.test.window.args.step_size={step_size}',
    ]
    return overrides
