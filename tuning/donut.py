from typing import List

import optuna


def generate_trial_overrides(trial: optuna.trial.Trial) -> List[str]:
    num_hidden_dimensions = trial.suggest_int('num_hidden_dimensions', 2, 4)
    hidden_dimension_size = trial.suggest_int('hidden_dimension_size', 50, 200, step=50)
    latent_dim = trial.suggest_int('latent_dim', 8, 64, step=8)
    mask_prob = trial.suggest_float('mask_prob', 0.0, 0.1)
    num_mc_samples = trial.suggest_int('num_mc_samples', 0, 64, step=16)
    window_size = trial.suggest_int('window_size', 0, 100, step=10)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)

    hidden_dimensions = [hidden_dimension_size] * num_hidden_dimensions
    num_mc_samples = 2 if num_mc_samples == 0 else num_mc_samples
    window_size = 5 if window_size == 0 else window_size

    overrides = [
        f'model_params.hidden_dims={hidden_dimensions}',
        f'model_params.latent_dim={latent_dim}',
        f'model_params.mask_prob={mask_prob}',
        f'transforms.train.window.args.window_size={window_size}',
        f'transforms.test.window.args.window_size={window_size}',
        f'transforms.seq_len={window_size}',
        f'run_params.optimizer.args.lr={learning_rate}',
        f'detector_params.num_mc_samples={num_mc_samples}',
    ]
    return overrides
