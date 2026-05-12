# House Prices — Kaggle Competition Solution

---

## Структура проекта

```
house_prices/
├── data/
│   ├── train.csv
│   └── test.csv
├── checkpoints/
│   ├── submission.csv          # финальные предсказания
│   ├── model_comparison.csv    # метрики всех моделей
│   └── ensemble_results.csv    # сравнение методов ансамблирования
├── models/
│   ├── __init__.py
│   ├── boosting.py             # CatBoost, LightGBM, XGBoost (регрессоры)
│   └── dnn.py                  # PyTorch MLP для регрессии и классификации
├── config.py                   # все настройки проекта
├── dataset.py                  # предобработка данных
├── solver.py                   # обучение, валидация, ансамбли
├── utils.py                    # вспомогательные функции
├── main.py                     # точка входа
└── requirements.txt
```

---

## Быстрый старт

**1. Установить зависимости**

```bash
pip install -r requirements.txt
```

**2. Запустить**

```bash
python main.py
```

После запуска в `checkpoints/` появятся:
- `submission.csv` — файл для загрузки на Kaggle
- `model_comparison.csv` — таблица метрик всех моделей
- `ensemble_results.csv` — сравнение методов ансамблирования

---

## Модели

Все модели включаются/выключаются в `config.py` через словарь `to_train`:

```python
to_train = {
    'ridge':    True,   # Ridge регрессия (L2)
    'lasso':    True,   # Lasso регрессия (L1)
    'elastic':  True,   # ElasticNet (L1 + L2)
    'knn':      True,   # K-Nearest Neighbors
    'dt':       True,   # Decision Tree
    'rf':       True,   # Random Forest
    'catboost': True,   # CatBoost
    'lgbm':     True,   # LightGBM
    'xgb':      True,   # XGBoost
    'dnn':      True,   # Deep Neural Network (PyTorch)
}
```

---

## Ансамбли

Режим ансамблирования задаётся в `config.py`:

```python
ensemble_mode = "all"   # "voting" | "weighted" | "stacking" | "all"
```

| Режим | Описание |
|---|---|
| `voting` | Среднее предсказаний всех моделей |
| `weighted` | Взвешенное среднее — лучшим моделям (меньший rmse) больший вес |
| `stacking` | Мета-модель (Ridge) обучается на OOF предсказаниях |
| `all` | Запускает все три и сравнивает результаты |

**Важно:** в режиме `all` финальный submission использует стекинг как наиболее мощный метод.

## Настройка параметров

Все параметры моделей задаются в `config.py`. Пример для CatBoost:

```python
catboost_params = {
    'iterations':    1000,
    'learning_rate': 0.05,
    'depth':         6,
    'loss_function': 'RMSE',
    'eval_metric':   'RMSE',
    'verbose':       0,
    'random_seed':   seed,
}
```

Параметры DNN:

```python
dnn_params = {
    'hidden_sizes':  [512, 256, 128],  # размеры скрытых слоёв
    'dropout':       0.3,
    'activation':    'relu',           # 'relu' | 'tanh' | 'selu'
    'epochs':        100,
    'batch_size':    64,
    'learning_rate': 1e-3,
    'weight_decay':  1e-4,
    'patience':      20,               # early stopping
    'scheduler':     'cosine',         # 'cosine' | 'step' | None
    'optimizer':     'AdamW',          # 'Adam' | 'AdamW' | 'SGD' | 'RMSprop'
    'device':        'cpu',            # автоматически 'cuda' если доступен GPU
}
```

---

## Логирование экспериментов

При каждом запуске результаты DNN автоматически дописываются в `checkpoints/model_comparison_experiments.csv`.

---

## Результаты

<pre>

Обучаем: xgb

  Fold 1: rmse_log=0.1328  rmse=25063
  Fold 2: rmse_log=0.1141  rmse=26814
  Fold 3: rmse_log=0.1569  rmse=40850
  Fold 4: rmse_log=0.1210  rmse=25588
  Fold 5: rmse_log=0.1026  rmse=19618

xgb ИТОГО
  rmse     = 27586.3149 ± 0.0185
  rmse_log = 0.1255
</pre>

...

Таблица результатов сохранена: checkpoints/model_comparison.csv

| Модель    | RMSE      | RMSE (log) | RMSE Std |
|-----------|-----------|------------|----------|
| CatBoost  | 27,002    | 0.1229     | 0.0163   |
| XGBoost   | 27,586    | 0.1255     | 0.0185   |
| LGBM      | 29,543    | 0.1333     | 0.0179   |
| RF        | 30,158    | 0.1389     | 0.0180   |
| Ridge     | 44,145    | 0.1440     | 0.0343   |
| Lasso     | 47,170    | 0.1445     | 0.0368   |
| ElasticNet| 48,696    | 0.1485     | 0.0335   |
| KNN       | 36,412    | 0.1687     | 0.0108   |
| DT        | 41,422    | 0.1954     | 0.0195   |
| DNN       | 73,335    | 0.3610     | 0.0442   |

...