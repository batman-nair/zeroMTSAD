from typing import List

import optuna


def generate_trial_overrides(trial: optuna.trial.Trial) -> List[str]:
    scheduler = trial.suggest_categorical(
        'scheduler',
        [
            {
                'class': 'torch.optim.lr_scheduler.LambdaLR',
                'args': {
                    'lr_lambda': 'lambda epoch: 0.95 ** epoch'
                },
            },
            {
                'class': 'torch.optim.lr_scheduler.StepLR',
                'args': {
                    'step_size': 100,
                    'gamma': 1
                }
            },
            {
                'class': 'torch.optim.lr_scheduler.MultiStepLR',
                'args': {
                    'milestones': [20],
                    'gamma': 0.1
                }
            },
            {
                'class': 'torch.optim.lr_scheduler.MultiStepLR',
                'args': {
                    'milestones': [30, 80],
                    'gamma': 0.1
                }
            },
            {
                'class': 'torch.optim.lr_scheduler.CosineAnnealingLR',
                'args': {
                    'T_max': 50
                }
            },
        ]

    )
    overrides = [
        f'run_params.scheduler.class={scheduler["class"]}',
        f'run_params.scheduler.args={scheduler["args"]}',
    ]

    return overrides

