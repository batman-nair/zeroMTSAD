import argparse

import matplotlib.pyplot as plt

import optuna
import optuna.visualization.matplotlib as optuna_vis

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=str, help='Experiment to be loaded', required=True)
    parser.add_argument('--db', type=str, help='Database path for optuna', default='sqlite:///optuna.db')
    args = parser.parse_args()

    study = optuna.load_study(study_name=args.study, storage=args.db)

    optuna_vis.plot_optimization_history(study)
    optuna_vis.plot_parallel_coordinate(study)
    optuna_vis.plot_slice(study)
    optuna_vis.plot_contour(study)
    optuna_vis.plot_param_importances(study)
    optuna_vis.plot_edf(study)

    print('Best parameters:', study.best_params)
    plt.show()
