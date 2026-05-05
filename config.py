import torch


class Config:
    seed = 42

    path_to_train       = 'data/train.csv'
    path_to_test        = 'data/test.csv'
    path_to_checkpoints = 'checkpoints/'
    path_to_submission  = 'checkpoints/submission.csv'

    target_col = 'SalePrice'
    id_col     = 'Id'

    n_splits = 5

    to_train = {
        'ridge':    True,
        'lasso':    True,
        'elastic':  True,
        'knn':      True,
        'dt':       True,
        'rf':       True,
        'catboost': True,
        'lgbm':     True,
        'xgb':      True,
        'dnn':      True,
    }

    ridge_params = {
        'alpha': 10.0,
    }

    lasso_params = {
        'alpha':    0.001,
        'max_iter': 5000,
    }

    elastic_params = {
        'alpha':   0.001,
        'l1_ratio': 0.5,
        'max_iter': 5000,
    }

    knn_params = {
        'n_neighbors': 7,
        'weights':     'distance',
        'metric':      'euclidean',
    }

    dt_params = {
        'max_depth':        5,
        'min_samples_split': 5,
        'random_state':     seed,
    }

    rf_params = {
        'n_estimators': 200,
        'max_depth':    None,
        'max_features': 'sqrt',
        'n_jobs':       -1,
        'random_state': seed,
    }

    catboost_params = {
        'iterations':    1000,
        'learning_rate': 0.05,
        'depth':         6,
        'loss_function': 'RMSE',
        'eval_metric':   'RMSE',
        'verbose':       0,
        'random_seed':   seed,
    }

    lgbm_params = {
        'n_estimators':  1000,
        'learning_rate': 0.05,
        'num_leaves':    31,
        'reg_alpha':     0.1,
        'reg_lambda':    0.1,
        'verbose':       -1,
        'random_state':  seed,
    }

    xgb_params = {
        'n_estimators':    1000,
        'learning_rate':   0.05,
        'max_depth':       4,
        'subsample':       0.8,
        'colsample_bytree': 0.8,
        'verbosity':       0,
        'random_state':    seed,
        'eval_metric':     'rmse',
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dnn_params = {
        'hidden_sizes':  [512, 256, 128],
        'dropout':       0.3,
        'activation':    'relu',
        'epochs':        100,
        'batch_size':    64,
        'learning_rate': 1e-3,
        'weight_decay':  1e-4,
        'patience':      20,
        'scheduler':     'cosine',
        'device':        device,
        'optimizer':     'AdamW',
    }

    meta_model_type   = 'ridge'
    meta_model_params = {'alpha': 10.0}

    ensemble_mode = 'all'   # 'voting' | 'weighted' | 'stacking' | 'all'
