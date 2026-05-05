import numpy as np
import pandas as pd

from config import Config
from utils import set_seed
from dataset import Dataset
from solver import Solver

import warnings
warnings.filterwarnings('ignore')


def run() -> dict:
    config = Config()
    set_seed(config.seed)

    df_train = pd.read_csv(config.path_to_train)
    df_test  = pd.read_csv(config.path_to_test)

    dataset = Dataset(config)
    X_train, X_test, y_train = dataset.get_dataset(df_train, df_test)

    solver = Solver(config)
    results = solver.fit(X_train, y_train)

    solver.fit_ensemble(X_train, y_train)
    preds_log = solver.predict_ensemble(X_test)

    # Переводим из log-пространства обратно в цены
    preds = np.expm1(preds_log)

    submission = pd.DataFrame({
        config.id_col:     df_test[config.id_col],
        config.target_col: preds,
    })
    submission.to_csv(config.path_to_submission, index=False)
    print(f'\nСабмит сохранён: {config.path_to_submission}')

    return results


if __name__ == '__main__':
    run()
