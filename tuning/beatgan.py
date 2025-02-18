from typing import List

import optuna


def generate_trial_overrides(trial: optuna.trial.Trial) -> List[str]:
    conv_filters = trial.suggest_int('conv_filters', 16, 64, step=16)
    latent_dim = trial.suggest_int('latent_dim', 25, 50, step=25)
    adversarial_weight = trial.suggest_float('adversarial_weight', 0.8, 1.0, step=0.1)
    distort_fraction = trial.suggest_float('distort_fraction', 0.05, 0.2, step=0.05)
    n_augmentations = trial.suggest_int('n_augmentations', 1, 3, step=1)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    # BeatGAN window size must be 320 according to the model

    overrides = [
        f'model_params.conv_filters={conv_filters}',
        f'model_params.latent_dim={latent_dim}',
        f'model_params.loss_params.adversarial_weight={adversarial_weight}',
        f'transforms.train.augmentation.args.distort_fraction={distort_fraction}',
        f'transforms.train.augmentation.args.n_augmentations={n_augmentations}',
        f'run_params.optimizer.args.lr={learning_rate}',
    ]
    return overrides

