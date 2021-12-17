# %% ------------------------------------------------------------------
# LOAD
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.tuned_selection import TimeSeriesSplit
from pycaret.regression import *

def eval_MAPE(y_true, y_pred):
    mape = np.abs(y_pred - y_true) / y_true
    return mape.mean()


# %% ------------------------------------------------------------------
# GENERATE FEATURES
df_6 = pd.read_pickle('data/df_6.pkl')
df = df_6.copy().sort_index()
n_period = df.index.levels[1].values.shape[0]
n_store = df.index.levels[0].values.shape[0]

df['period'] = np.tile(range(n_period), n_store)
df['year'] = np.tile(df.index.levels[1].year, n_store)
df['month'] = np.tile(df.index.levels[1].month, n_store)

# promo: 1 if Jan or Nov (biggest sale), 0 otherwise
df['promo'] = np.where((df['month'] == 1) | (df['month'] == 11), True, False)


# group by store
df_grouped = df.groupby(level='store')
df_grouped.get_group('LONG AN').tail(14)

# lags
df['lag_14'] = df_grouped.shift(14)['sales']

# rolling moving averages
df['mva_2'] = df_grouped.rolling(2).mean()['lag_14'].values
df['mva_3'] = df_grouped.rolling(3).mean()['lag_14'].values
df['mva_4'] = df_grouped.rolling(4).mean()['lag_14'].values
# expanding moving averages
df['ex_mva_2'] = df_grouped.expanding(min_periods=2).mean()['lag_14'].values
df['ex_mva_3'] = df_grouped.expanding(min_periods=3).mean()['lag_14'].values
df['ex_mva_4'] = df_grouped.expanding(min_periods=4).mean()['lag_14'].values
# exponential weighted moving averages
df['ew_mva_2'] = df_grouped.ewm(span=2, min_periods=2).mean()['lag_14'].values
df['ew_mva_3'] = df_grouped.ewm(span=3, min_periods=2).mean()['lag_14'].values
df['ew_mva_4'] = df_grouped.ewm(span=4, min_periods=2).mean()['lag_14'].values

# drop NAs
df = df.dropna()

# arrange columns by type for cat/num transformation later
df = df[['sales',
         'store_group',
         'store_format',
         'store_level',
         'store_segment',
         'province',
         'year',
         'month',
         'promo',
         'store_area',
         'number_of_staff',
         'holiday_neg',
         'lag_14',
         'mva_2',
         'mva_3',
         'mva_4',
         'ex_mva_2',
         'ex_mva_3',
         'ex_mva_4',
         'ew_mva_2',
         'ew_mva_3',
         'ew_mva_4']]


# train/test split
df_train = df.reset_index('store').sort_index()[:'2020-12']
df_test = df.reset_index('store').sort_index()['2021-01':]

# check data shape
df_train.shape, df_test.shape


# %% -----------------------------------
# BENCHMARK
y_train = df_train['sales']
y_test = df_test['sales']

fc_naive = np.repeat(y_train[-1:].values, len(y_test))
mape_naive = eval_MAPE(y_test, fc_naive)

fc_mean = np.repeat(y_train.mean(), len(y_test))
mape_mean = eval_MAPE(y_test, fc_mean)

fc_median = np.repeat(y_train.median(), len(y_test))
mape_median = eval_MAPE(y_test, fc_median)

print('''Benchmark MAPE:\n
Naive:  {:.3}
Mean:   {:.3}
Median: {:.3}'''
        .format(mape_naive, mape_mean, mape_median))

# %% -----------------------------------------------------------
# intialize the setup

categorical_features = [
    'store',
    'store_group',
    'store_format',
    'store_segment',
    'province',
    'year',
    'month',
    'promo'
]
numeric_features = [
    'store_area',
    'number_of_staff',
    'holiday_neg',
    'lag_14',
    'mva_2',
    'mva_3',
    'mva_4',
    'ex_mva_2',
    'ex_mva_3',
    'ex_mva_4',
    'ew_mva_2',
    'ew_mva_3',
    'ew_mva_4'
]
ordinal_features = {
    'store_level': ['C', 'B', 'A', 'A++']
}

ini_setup = setup(data=df_train,
                  target='sales',
                  numeric_features=numeric_features,
                  categorical_features=categorical_features,
                  ordinal_features=ordinal_features,
                  
                  # transformation steps
                  normalize=True,
                #   pca=True,
                  # interactive features???
                  
                  # CV
                  data_split_shuffle=False,
                  fold_strategy=TimeSeriesSplit(4, test_size=13),
                  
                  use_gpu=True,
                  session_id=3,
                  silent=True
                  )

# %% -----------------------------------------------------------
# spot check all available models
best_model = compare_models(sort='MAPE',
                            turbo=False, # include all models
                            round=3,
                            n_select=1,
                            verbose=False)  # top 1

# %% -----------------------------------------------------------
# hp tuning

def MAPE_tuned(unseen_data, model):
    tuned_model = tune_model(model, optimize='MAPE', verbose=False)
    pred_unseen = predict_model(tuned_model, unseen_data)
    MAPE_unseen = eval_MAPE(pred_unseen['sales'], pred_unseen['Label'])
    return tuned_model, MAPE_unseen

tuned_1, MAPE_unseen_1 = MAPE_tuned(df_test, best_model[0])
MAPE_unseen_1

tuned_2, MAPE_unseen_2 = MAPE_tuned(df_test, best_model[1])
tuned_3, MAPE_unseen_3 = MAPE_tuned(df_test, best_model[2])

# %% summary
all_store_score = pd.DataFrame(
    {'model': [type(tuned_1).__name__,
               type(tuned_2).__name__,
               type(tuned_3).__name__,
               ],
        'MAPE': [round(MAPE_unseen_1, 3),
                 round(MAPE_unseen_2, 3),
                 round(MAPE_unseen_3, 3),
                 ],
     }).sort_values('MAPE').reset_index(drop=True)

all_store_score

# %% 
all_store_score.to_csv('pycaret_all_store_weekly.csv')

#%%
fea_imp = tuned_1.feature_importances_
trans_fea = get_config('X').columns.values
fea_imp = pd.Series(fea_imp, index=trans_fea).sort_values(ascending=False)
fea_imp.head()
