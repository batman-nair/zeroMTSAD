from typing import List

import optuna


def generate_trial_overrides(trail: optuna.trial.Trial) -> List[str]:
    hidden_dimensions = trail.suggest_categorical(
        'hidden_dimensions',
        [
            "[30]",
            "[40]",
            "[30, 30]",
            "[50, 50]"
        ])
    window_size = trail.suggest_int('window_size', 5, 100)
    learning_rate = trail.suggest_float('learning_rate', 1e-5, 1e-1, log=True)

    overrides = [
        f'model_params.hidden_dimensions={hidden_dimensions}',
        f'transforms.train.window.args.window_size={window_size}',
        f'transforms.test.window.args.window_size={window_size}',
        f'transforms.seq_len={window_size}',
        f'run_params.optimizer.args.lr={learning_rate}',
    ]
    return overrides
