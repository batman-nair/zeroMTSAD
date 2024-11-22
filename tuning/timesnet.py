from typing import List

import optuna


def generate_trial_overrides(trial: optuna.trial.Trial) -> List[str]:
    top_k = trial.suggest_int('top_k', 5, 20, step=2)
    d_model = trial.suggest_int('d_model', 32, 128, step=32)
    d_ff = trial.suggest_int('d_ff', 32, 128, step=32)
    num_kernels = trial.suggest_int('num_kernels', 4, 16, step=4)
    e_layers = trial.suggest_int('e_layers', 2, 6, step=2)
    dropout = trial.suggest_float('dropout', 0.0, 0.2, step=0.1)
    window_size = trial.suggest_int('window_size', 25, 100, step=25)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)

    overrides = [
        f'model_params.top_k={top_k}',
        f'model_params.d_model={d_model}',
        f'model_params.d_ff={d_ff}',
        f'model_params.num_kernels={num_kernels}',
        f'model_params.e_layers={e_layers}',
        f'model_params.dropout={dropout}',
        f'transforms.train.window.args.window_size={window_size}',
        f'transforms.test.window.args.window_size={window_size}',
        f'transforms.seq_len={window_size}',
        f'run_params.optimizer.args.lr={learning_rate}',
    ]
    return overrides
