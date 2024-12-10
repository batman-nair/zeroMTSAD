from typing import List

import optuna


def generate_trial_overrides(trial: optuna.trial.Trial) -> List[str]:
    moving_avg = trial.suggest_int('moving_avg', 15, 55, step=10)
    model_dim = trial.suggest_int('model_dim', 64, 512, step=64)
    dropout = trial.suggest_float('dropout', 0.0, 0.2, step=0.1)
    num_heads = trial.suggest_int('num_heads', 4, 16, step=4)
    fcn_dim = trial.suggest_int('fcn_dim', 64, 512, step=64)
    encoder_layers = trial.suggest_int('encoder_layers', 2, 6, step=1)
    modes = trial.suggest_int('modes', 16, 64, step=16)
    window_size = trial.suggest_int('window_size', 25, 100, step=25)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)

    overrides = [
        f'model_params.moving_avg={moving_avg}',
        f'model_params.model_dim={model_dim}',
        f'model_params.dropout={dropout}',
        f'model_params.num_heads={num_heads}',
        f'model_params.fcn_dim={fcn_dim}',
        f'model_params.encoder_layers={encoder_layers}',
        f'model_params.modes={modes}',
        f'transforms.train.window.args.window_size={window_size}',
        f'transforms.test.window.args.window_size={window_size}',
        f'transforms.seq_len={window_size}',
        f'run_params.optimizer.args.lr={learning_rate}',
    ]
    return overrides
