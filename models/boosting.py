import pandas as pd
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb


def train_catboost(X_train, y_train, X_val, y_val, params):
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    return model


def train_lgbm(X_train, y_train, X_val, y_val, params):
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    return model


def train_xgb(X_train, y_train, X_val, y_val, params):
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model
