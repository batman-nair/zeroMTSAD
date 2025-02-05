from typing import List

import optuna


def generate_trial_overrides(trial: optuna.trial.Trial) -> List[str]:
    hidden_dimensions = [
        [30],
        [50],
        [30, 30],
        [50, 50],
    ]
    lstm_hidden_dim_size = trial.suggest_int('lstm_hidden_dim_size', 0, len(hidden_dimensions) - 1)
    linear_hidden_dim_size = trial.suggest_int('linear_hidden_dim_size', 0, len(hidden_dimensions) - 1)
    window_size = trial.suggest_int('window_size', 5, 100, step=5)
    prediction_horizon = trial.suggest_int('prediction_horizon', 2, 10, step=2)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)

    overrides = [
        f'model_params.lstm_hidden_dims={hidden_dimensions[lstm_hidden_dim_size]}',
        f'model_params.linear_hidden_layers={hidden_dimensions[linear_hidden_dim_size]}',
        f'model_params.prediction_horizon={prediction_horizon}',
        f'transforms.train.prediction.args.window_size={window_size}',
        f'transforms.train.prediction.args.prediction_horizon={prediction_horizon}',
        f'transforms.test.prediction.args.window_size={window_size}',
        f'transforms.test.prediction.args.prediction_horizon={prediction_horizon}',
        f'transforms.seq_len={window_size}',
        f'run_params.optimizer.args.lr={learning_rate}',
    ]
    return overrides
