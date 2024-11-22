from typing import List

import optuna


def generate_trial_overrides(trial: optuna.trial.Trial) -> List[str]:
    hidden_dimensions = [
        [30],
        [50],
        [30, 30],
        [50, 50],
        [30, 30, 30],
        [50, 50, 50],
    ]
    hidden_dimensions_size = trial.suggest_int('hidden_dimensions_size', 0, len(hidden_dimensions) - 1)
    window_size = trial.suggest_int('window_size', 5, 100, step=5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)

    overrides = [
        f'model_params.hidden_dimensions={hidden_dimensions[hidden_dimensions_size]}',
        f'transforms.train.window.args.window_size={window_size}',
        f'transforms.test.window.args.window_size={window_size}',
        f'transforms.seq_len={window_size}',
        f'run_params.optimizer.args.lr={learning_rate}',
    ]
    return overrides
