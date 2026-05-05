import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import LabelEncoder
from config import Config


class Dataset:

    def __init__(self, config: Config) -> None:
        self.config = config
        self.data_checkpoint = {}

    def get_dataset(
        self,
        df_train: pd.DataFrame,
        df_test:  pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:

        y_train = np.log1p(df_train[self.config.target_col].copy())

        df_train = self._fill_missing(df_train, fit=True)
        df_test  = self._fill_missing(df_test,  fit=False)

        df_train = self._feature_engineering(df_train)
        df_test  = self._feature_engineering(df_test)

        df_train, df_test = self._encode_categoricals(df_train, df_test)

        drop = [self.config.target_col, self.config.id_col]
        X_train = df_train.drop(columns=[c for c in drop if c in df_train.columns])
        X_test  = df_test.drop(columns=[c for c in drop if c in df_test.columns])

        # Выравниваем колонки test по train
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        print(f'X_train: {X_train.shape} | X_test: {X_test.shape}')
        assert X_train.isnull().sum().sum() == 0, 'В X_train остались пропуски!'
        assert X_test.isnull().sum().sum()  == 0, 'В X_test остались пропуски!'

        return X_train, X_test, y_train

    def _fill_missing(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:

        df = df.copy()

        # Категориальные где NA = реальное значение "нет объекта"
        none_cols = [
            'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
            'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
            'BsmtFinType2', 'MasVnrType',
        ]
        for col in none_cols:
            if col in df.columns:
                df[col] = df[col].fillna('None')

        # Числовые где NA = реальный ноль (нет гаража, подвала и т.д.)
        zero_cols = [
            'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
            'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',
            'GarageArea', 'GarageCars',
        ]
        for col in zero_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # GarageYrBlt — заполняем YearBuilt (гараж строился с домом)
        if 'GarageYrBlt' in df.columns:
            df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])

        # LotFrontage — медиана по Neighborhood (соседи имеют похожие участки)
        if 'LotFrontage' in df.columns:
            if fit:
                self.data_checkpoint['lotfrontage_medians'] = (
                    df.groupby('Neighborhood')['LotFrontage'].median()
                )
            medians = self.data_checkpoint['lotfrontage_medians']
            mask = df['LotFrontage'].isna()
            df.loc[mask, 'LotFrontage'] = df.loc[mask, 'Neighborhood'].map(medians)
            df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

        # Electrical — мода
        if 'Electrical' in df.columns:
            if fit:
                self.data_checkpoint['electrical_mode'] = df['Electrical'].mode()[0]
            df['Electrical'] = df['Electrical'].fillna(
                self.data_checkpoint['electrical_mode']
            )

        # Остальные категориальные — мода
        cat_cols = df.select_dtypes(include=['object', 'str']).columns
        for col in cat_cols:
            if df[col].isnull().any():
                if fit:
                    self.data_checkpoint[f'mode_{col}'] = df[col].mode()[0]
                df[col] = df[col].fillna(self.data_checkpoint.get(f'mode_{col}', 'Unknown'))

        # Остальные числовые — медиана по train
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if df[col].isnull().any():
                if fit:
                    self.data_checkpoint[f'median_{col}'] = df[col].median()
                df[col] = df[col].fillna(self.data_checkpoint.get(f'median_{col}', 0))

        return df

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Общая площадь
        df['TotalSF']     = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
        df['TotalPorchSF'] = (
            df['OpenPorchSF'] + df['EnclosedPorch'] +
            df['3SsnPorch']   + df['ScreenPorch']
        )

        # Возраст и ремонт
        df['HouseAge']    = df['YrSold'] - df['YearBuilt']
        df['RemodAge']    = df['YrSold'] - df['YearRemodAdd']
        df['IsRemodeled'] = (df['YearBuilt'] != df['YearRemodAdd']).astype(int)
        df['IsNew']       = (df['YrSold'] == df['YearBuilt']).astype(int)

        # Ванные
        df['TotalBath'] = (
            df['FullBath'] + 0.5 * df['HalfBath'] +
            df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
        )

        # Есть ли объект (бинарные флаги)
        df['HasPool']     = (df['PoolArea'] > 0).astype(int)
        df['HasGarage']   = (df['GarageArea'] > 0).astype(int)
        df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
        df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
        df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)

        # Качество × площадь
        df['QualArea'] = df['OverallQual'] * df['GrLivArea']

        # Log-трансформация признаков с сильной асимметрией
        skewed_cols = [
            'LotArea', 'GrLivArea', 'TotalBsmtSF', 'TotalSF',
            '1stFlrSF', 'GarageArea', 'LotFrontage',
        ]
        for col in skewed_cols:
            if col in df.columns:
                df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))

        return df

    def _encode_categoricals(
        self,
        df_train: pd.DataFrame,
        df_test:  pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ordinal_maps = {
            'ExterQual':   {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'ExterCond':   {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'BsmtQual':    {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'BsmtCond':    {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'BsmtExposure':{'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
            'BsmtFinType1':{'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
            'BsmtFinType2':{'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
            'HeatingQC':   {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'FireplaceQu': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'GarageFinish':{'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},
            'GarageQual':  {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'GarageCond':  {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'PoolQC':      {'None': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
            'Fence':       {'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4},
            'LotShape':    {'IR3': 1, 'IR2': 2, 'IR1': 3, 'Reg': 4},
            'LandSlope':   {'Sev': 1, 'Mod': 2, 'Gtl': 3},
            'PavedDrive':  {'N': 0, 'P': 1, 'Y': 2},
            'Functional':  {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4,
                            'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8},
        }

        for col, mapping in ordinal_maps.items():
            if col in df_train.columns:
                df_train[col] = df_train[col].map(mapping).fillna(0).astype(int)
                df_test[col]  = df_test[col].map(mapping).fillna(0).astype(int)

        cat_cols = df_train.select_dtypes(include=['object', 'str']).columns.tolist()
        for col in cat_cols:
            le = LabelEncoder()
            le.fit(
                pd.concat([df_train[col], df_test[col]], axis=0).astype(str)
            )
            df_train[col] = le.transform(df_train[col].astype(str))
            df_test[col]  = le.transform(df_test[col].astype(str))

        return df_train, df_test
