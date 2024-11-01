from typing import List

import optuna


def generate_trial_overrides(trial: optuna.trial.Trial) -> List[str]:
    k = trial.suggest_int('k', 1, 101, step=5)
    batch_size = trial.suggest_int('batch_size', 512, 10240, step=512)
    window_size = trial.suggest_int('window_size', 5, 100, step=5)
    step_size = trial.suggest_int('step_size', 1, 10, step=1)

    overrides = [
        f'batch_size={batch_size}',
        f'model_params.k={k}',
        f'model_params.batch_size={batch_size}',
        f'transforms.train.window.args.window_size={window_size}',
        f'transforms.test.window.args.window_size={window_size}',
        f'transforms.seq_len={window_size}',
        f'transforms.train.window.args.step_size={step_size}',
        f'transforms.test.window.args.step_size={step_size}',
    ]
    return overrides
