from typing import List

import optuna


def generate_trial_overrides(trial: optuna.trial.Trial) -> List[str]:
    gru_hidden_dim = trial.suggest_int('gru_hidden_dim', 32, 256, step=32)
    latent_dim = trial.suggest_int('latent_dim', 8, 32, step=8)
    gmm_components = trial.suggest_int('gmm_components', 2, 10)
    num_mc_samples = trial.suggest_int('num_mc_samples', 2, 64, step=2)
    window_size = trial.suggest_int('window_size', 5, 100, step=5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)

    overrides = [
        f'model_params.gru_hidden_dims=[{gru_hidden_dim}]',
        f'model_params.latent_dim={latent_dim}',
        f'model_params.gmm_components={gmm_components}',
        f'transforms.train.window.args.window_size={window_size}',
        f'transforms.test.window.args.window_size={window_size}',
        f'transforms.seq_len={window_size}',
        f'run_params.optimizer.args.lr={learning_rate}',
        f'detector_params.num_mc_samples={num_mc_samples}',
    ]
    return overrides
