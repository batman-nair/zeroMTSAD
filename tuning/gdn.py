from typing import List

import optuna


def generate_trial_overrides(trial: optuna.trial.Trial) -> List[str]:
    dim = trial.suggest_int('dim', 32, 128, step=32)
    out_layer_hidden_dim = trial.suggest_int('out_layer_hidden_dim', 32, 128, step=32)
    topk = trial.suggest_int('topk', 15, 30, step=5)
    dropout = trial.suggest_float('dropout', 0.0, 0.4, step=0.1)
    window_size = trial.suggest_int('window_size', 5, 100, step=5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)

    overrides = [
        f'model_params.dim={dim}',
        f'model_params.out_layer_hidden_dims=[{out_layer_hidden_dim}, {out_layer_hidden_dim}]',
        f'model_params.topk={topk}',
        f'model_params.dropout_prob={dropout}',
        f'transforms.train.prediction.args.window_size={window_size}',
        f'transforms.test.prediction.args.window_size={window_size}',
        f'transforms.seq_len={window_size}',
        f'run_params.optimizer.args.lr={learning_rate}',
    ]
    return overrides
