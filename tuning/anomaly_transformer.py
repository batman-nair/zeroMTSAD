from typing import List

import optuna


def generate_trial_overrides(trial: optuna.trial.Trial) -> List[str]:
    d_model = trial.suggest_int('d_model', 64, 512, step=64)
    n_heads = trial.suggest_int('n_heads', 4, 16, step=2)
    e_layers = trial.suggest_int('e_layers', 2, 6, step=2)
    d_ff = trial.suggest_int('d_ff', 64, 512, step=64)
    dropout = trial.suggest_float('dropout', 0.0, 0.4)
    window_size = trial.suggest_int('window_size', 5, 100, step=5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)

    overrides = [
        f'model_params.d_model={d_model}',
        f'model_params.n_heads={n_heads}',
        f'model_params.e_layers={e_layers}',
        f'model_params.d_ff={d_ff}',
        f'model_params.dropout={dropout}',
        f'transforms.train.window.args.window_size={window_size}',
        f'transforms.test.window.args.window_size={window_size}',
        f'transforms.seq_len={window_size}',
        f'run_params.optimizer.args.lr={learning_rate}',
    ]
    return overrides
