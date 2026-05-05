import os
import csv
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rmse(y_true, y_pred) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))


def print_scores(model_name: str, scores: dict) -> None:
    print(f'\n{model_name}')
    print(f'  rmse     = {scores["rmse"]:.4f} ± {scores["rmse_std"]:.4f}')
    print(f'  rmse_log = {scores["rmse_log"]:.4f}')


def save_results_table(all_results: dict, path: str, extra_params: dict = None) -> None:
    rows = []
    for name, res in all_results.items():
        rows.append({
            'model':    name,
            'rmse':     round(res['rmse'],     4),
            'rmse_log': round(res['rmse_log'], 4),
            'rmse_std': round(res['rmse_std'], 4),
        })
    df = pd.DataFrame(rows).sort_values('rmse_log')
    df.to_csv(path, index=False)
    print(f'\nТаблица результатов сохранена: {path}')
    print(df.to_string(index=False))

    if extra_params is not None:
        log_path = path.replace('.csv', '_experiments.csv')
        dnn_scores = all_results.get('dnn', {})
        row = {
            **{k: v for k, v in extra_params.items() if k != 'device'},
            'rmse':     round(dnn_scores.get('rmse',     0), 4),
            'rmse_log': round(dnn_scores.get('rmse_log', 0), 4),
        }
        file_exists = os.path.exists(log_path)
        with open(log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        print(f'Лог эксперимента дописан: {log_path}')


def log_ensemble_results(results: dict, path: str) -> None:
    rows = [{'ensemble': k, 'rmse_log': round(v['rmse_log'], 4)}
            for k, v in results.items()]
    df = pd.DataFrame(rows).sort_values('rmse_log')
    df.to_csv(path, index=False)
    print(f'\nРезультаты ансамблей сохранены: {path}')


def evaluate_ensembles(y_true, preds_dict: dict) -> dict:
    results = {}
    for name, preds in preds_dict.items():
        score = rmse(y_true, preds)
        results[name] = {'rmse_log': score}
    return results
