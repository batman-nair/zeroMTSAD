import os
import json
import pandas as pd
import matplotlib.pyplot as plt

def load_results(base_dir: str) -> pd.DataFrame:
    '''
    Load results, run configs, and scores from the logs folder.
    '''
    data = []
    for exp_dir in os.listdir(base_dir):
        exp_path = os.path.join(base_dir, exp_dir)
        if not os.path.isdir(exp_path):
            continue
        for version_dir in os.listdir(exp_path):
            version_path = os.path.join(exp_path, version_dir)
            if not os.path.isdir(version_path):
                continue

            results_path = os.path.join(version_path, 'results.json')
            config_path = os.path.join(version_path, 'config.json')
            run_path = os.path.join(version_path, 'run.json')

            if (not os.path.exists(results_path) or
                not os.path.exists(config_path) or
                not os.path.exists(run_path)):
                print(f'Skipping {version_path}, missing files')
                continue

            with open(results_path, 'r') as ff:
                results = json.load(ff)
            with open(config_path, 'r') as ff:
                config = json.load(ff)
            with open(run_path, 'r') as ff:
                run = json.load(ff)

            for metric, value_info in results.items():
                data.append({
                    'experiment': exp_dir,
                    'version': version_dir,
                    'metric': metric,
                    'value': value_info['score'],
                    'config': config,
                    'run_info': run,
                })

    return pd.DataFrame(data)

def filter_runs(data: pd.DataFrame, config_filters: dict) -> pd.DataFrame:
    '''
    Filter the runs based on the config filters.
    '''
    mask = None
    for key, value in config_filters.items():
        if mask is None:
            mask = data['config'].apply(lambda x: x.get(key) == value)
        else:
            mask &= data['config'].apply(lambda x: x.get(key) == value)

    return data[mask]

def get_nested_value(d, key_path):
    keys = key_path.split('.')
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return None
    return d

def filter_runs(data: pd.DataFrame, metric_filter: str, config_filters: dict) -> pd.DataFrame:
    """
    Filter the runs based on nested configuration options.
    For nested filters (e.g., "data_params.test_server_ids"), use a dotted key path.
    """

    def config_matches(row, filters):
        for key_path, expected_value in filters.items():
            actual_value = get_nested_value(row.get('config', {}), key_path)
            if actual_value != expected_value:
                return False
        return True

    mask = data.apply(
        lambda row: (
            config_matches(row, config_filters) and
            row['metric'] == metric_filter
        ),
        axis=1
    )
    return data[mask]

def extract_data(data: pd.DataFrame, config_keys: list, metric: str) -> pd.DataFrame:
    """
    Extract data as <set of config values> : <metric_score>.
    """
    extracted_data = []
    for _, row in data.iterrows():
        if row['metric'] == metric:
            # Extract the desired configuration values
            config_values = {key: get_nested_value(row['config'], key) for key in config_keys}
            config_values['score'] = row['value']
            extracted_data.append(config_values)

    return pd.DataFrame(extracted_data)

def extract_results(
        data: pd.DataFrame,
        dataset: str,
        type: str,
        test_ids: list,
        metric: str='best_ts_f1_score'
        ) -> pd.DataFrame:
    """
    Extract results for a specific dataset, type, and test dataset ids.
    Type is one of ['normal', 'zero']
    """
    if type not in ['normal', 'zero']:
        raise ValueError(f'Invalid type: {type}')
    if len(test_ids) > 1:
        final_data = None
        for test_id in test_ids:
            part_data = extract_results(data, dataset, type, [test_id], metric)
            if final_data is None:
                final_data = part_data
            else:
                final_data = pd.concat([final_data, part_data])
        return final_data
    if type == 'normal':
        train_ids = test_ids
    elif type == 'zero':
        if dataset == 'smd':
            train_ids = list(range(0, 20))
        elif dataset == 'exathlon':
            train_ids = list(range(1, 5))
        else:
            NotImplementedError(f'Unknown dataset: {dataset}')
    filters = {
        'dataset': dataset,
        'data_params.train_ids': train_ids,
        'data_params.test_ids': test_ids,
    }
    old_filters = {
        'dataset': dataset,
        'data_params.train_server_ids': train_ids,
        'data_params.test_server_ids': test_ids,
    }
    filtered_data = filter_runs(data, metric, filters)
    old_filtered_data = filter_runs(data, metric, old_filters)
    filtered_data = pd.concat([filtered_data, old_filtered_data])
    extracted_data = extract_data(filtered_data, ['experiment'], metric)
    return extracted_data



if __name__ == '__main__':
    logs_dir = 'final_logs'

    df = load_results(logs_dir)
    metric_choice = 'best_ts_f1_score'
    filtered_data = filter_runs(
        df,
        metric_choice,
        {
            'data_params.val_server_ids': [26],
            'data_params.test_server_ids': [26],
        }
    )

    extracted_data = extract_data(
        filtered_data,
        ['experiment'],
        metric_choice
    )

    plt.figure()
    plt.scatter(extracted_data['experiment'], extracted_data['score'])
    plt.xlabel('Experiment')
    plt.ylabel('Score')
    plt.show()
