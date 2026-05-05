import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from config import Config
from utils import print_scores, save_results_table, rmse, log_ensemble_results, evaluate_ensembles
from models.boosting import train_catboost, train_lgbm, train_xgb
from models.dnn import DNNClassifier


NEEDS_SCALING = {'ridge', 'lasso', 'elastic', 'knn', 'dnn'}


class Solver:

    def __init__(self, config: Config) -> None:
        self.config     = config
        self.models     = {}
        self.scalers    = {}
        self.results    = {}
        self.meta_model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        os.makedirs(self.config.path_to_checkpoints, exist_ok=True)

        for model_name, should_train in self.config.to_train.items():
            if not should_train:
                continue
            print(f'\n{"="*50}')
            print(f'Обучаем: {model_name}')
            print('='*50)
            self._train_one(X, y, model_name)

        save_results_table(
            self.results,
            os.path.join(self.config.path_to_checkpoints, 'model_comparison.csv'),
            extra_params=self.config.dnn_params,
        )
        return self.results

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.mean(list(self._get_all_probas(X).values()), axis=0)

    def predict_weighted(self, X: pd.DataFrame) -> np.ndarray:
        all_preds = self._get_all_probas(X)

        errors = {name: self.results[name]['rmse_log'] for name in all_preds}
        inv    = {name: 1.0 / (e + 1e-9) for name, e in errors.items()}
        total  = sum(inv.values())
        weights = {name: v / total for name, v in inv.items()}

        assert abs(sum(weights.values()) - 1.0) < 1e-6
        
        return sum(all_preds[name] * weights[name] for name in all_preds)

    def fit_stacking(self, X: pd.DataFrame, y: pd.Series) -> None:
        assert len(self.results) > 0, 'Сначала вызови fit()'

        stacking_features = pd.DataFrame({
            name: self.results[name]['oof_preds']
            for name in self.results
        }, index=X.index)

        if self.config.meta_model_type == 'ridge':
            meta = Ridge(**self.config.meta_model_params)
        else:
            raise ValueError(f'Неизвестный тип мета-модели: {self.config.meta_model_type}')


        # CV оценка мета-модели
        kf = KFold(n_splits=self.config.n_splits, shuffle=True, random_state=self.config.seed)
        cv_scores = []
        for tr_idx, val_idx in kf.split(stacking_features):
            meta.fit(stacking_features.iloc[tr_idx], y.iloc[tr_idx])
            preds = meta.predict(stacking_features.iloc[val_idx])
            cv_scores.append(rmse(y.iloc[val_idx], preds))

        print(f'\nStacking CV rmse_log: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}')

        meta.fit(stacking_features, y)
        self.meta_model = meta

    def predict_stacking(self, X: pd.DataFrame) -> np.ndarray:
        assert self.meta_model is not None, 'Сначала вызови fit_stacking()'

        test_features = pd.DataFrame(
            {name: preds for name, preds in self._get_all_probas(X).items()},
            index=X.index,
        )
        return self.meta_model.predict(test_features)

    def fit_ensemble(self, X: pd.DataFrame, y: pd.Series) -> None:
        if self.config.ensemble_mode in ('stacking', 'all'):
            self.fit_stacking(X, y)

        if self.config.ensemble_mode == 'all':
            preds_dict = {
                'voting':   self.predict(X),
                'weighted': self.predict_weighted(X),
                'stacking': self.predict_stacking(X),
            }
            ensemble_scores = evaluate_ensembles(y, preds_dict)
            log_ensemble_results(
                ensemble_scores,
                os.path.join(self.config.path_to_checkpoints, 'ensemble_results.csv'),
            )

    def predict_ensemble(self, X: pd.DataFrame) -> np.ndarray:
        mode = self.config.ensemble_mode
        if mode == 'voting':
            return self.predict(X)
        elif mode == 'weighted':
            return self.predict_weighted(X)
        elif mode == 'stacking':
            return self.predict_stacking(X)
        elif mode == 'all':
            return self.predict_stacking(X)
        else:
            raise ValueError(f'Unknown ensemble_mode: {mode}')

    def _get_all_probas(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        all_preds = {}
        for model_name, fold_models in self.models.items():
            fold_preds = []
            for fold_idx, model in enumerate(fold_models):
                X_input = X.copy()

                if model_name in NEEDS_SCALING:
                    scaler  = self.scalers[model_name][fold_idx]
                    X_input = pd.DataFrame(
                        scaler.transform(X_input), columns=X.columns
                    )
                preds = model.predict(X_input)
                fold_preds.append(preds)
            all_preds[model_name] = np.mean(fold_preds, axis=0)
        return all_preds

    def _build_model(self, model_name: str):
        cfg = self.config
        builders = {
            'ridge':   lambda: Ridge(**cfg.ridge_params),
            'lasso':   lambda: Lasso(**cfg.lasso_params),
            'elastic': lambda: ElasticNet(**cfg.elastic_params),
            'knn':     lambda: KNeighborsRegressor(**cfg.knn_params),
            'dt':      lambda: DecisionTreeRegressor(**cfg.dt_params),
            'rf':      lambda: RandomForestRegressor(**cfg.rf_params),
            'dnn':     lambda: DNNClassifier(**cfg.dnn_params),
        }
        return builders[model_name]() if model_name in builders else None

    def _train_fold(
        self,
        model_name: str,
        X_tr:  pd.DataFrame, y_tr:  pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series,
    ) -> Tuple:
        scaler = None

        if model_name in NEEDS_SCALING:
            scaler = StandardScaler()
            X_tr  = pd.DataFrame(scaler.fit_transform(X_tr),  columns=X_tr.columns)
            X_val = pd.DataFrame(scaler.transform(X_val),     columns=X_val.columns)

        if model_name == 'catboost':
            model = train_catboost(X_tr, y_tr, X_val, y_val, self.config.catboost_params)
        elif model_name == 'lgbm':
            model = train_lgbm(X_tr, y_tr, X_val, y_val, self.config.lgbm_params)
        elif model_name == 'xgb':
            model = train_xgb(X_tr, y_tr, X_val, y_val, self.config.xgb_params)
        elif model_name == 'dnn':
            model = self._build_model('dnn')
            model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
        else:
            model = self._build_model(model_name)
            model.fit(X_tr, y_tr)

        return model, scaler

    def _train_one(self, X: pd.DataFrame, y: pd.Series, model_name: str) -> None:
        if model_name == 'dnn':
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.config.seed)

        kf = KFold(
            n_splits=self.config.n_splits,
            shuffle=True,
            random_state=self.config.seed,
        )

        fold_models, fold_scalers = [], []
        rmse_list, rmse_log_list  = [], []
        oof_preds = np.zeros(len(X))

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr  = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_tr  = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            model, scaler = self._train_fold(model_name, X_tr, y_tr, X_val, y_val)

            X_val_input = X_val.copy()
            if scaler is not None:
                X_val_input = pd.DataFrame(
                    scaler.transform(X_val_input), columns=X.columns
                )

            preds = model.predict(X_val_input)
            oof_preds[val_idx] = preds

            # Метрика в log-пространстве (y уже log1p)
            rmse_log = rmse(y_val, preds)
            # Метрика в оригинальном пространстве
            rmse_orig = rmse(np.expm1(y_val), np.expm1(preds))

            rmse_log_list.append(rmse_log)
            rmse_list.append(rmse_orig)

            fold_models.append(model)
            fold_scalers.append(scaler)

            print(f'  Fold {fold+1}: rmse_log={rmse_log:.4f}  rmse={rmse_orig:.0f}')

        self.models[model_name]  = fold_models
        self.scalers[model_name] = fold_scalers
        self.results[model_name] = {
            'rmse':      np.mean(rmse_list),
            'rmse_log':  np.mean(rmse_log_list),
            'rmse_std':  np.std(rmse_log_list),
            'oof_preds': oof_preds,
        }

        scores_to_print = {k: v for k, v in self.results[model_name].items()
                           if k != 'oof_preds'}
        print_scores(f'{model_name} ИТОГО', scores_to_print)
