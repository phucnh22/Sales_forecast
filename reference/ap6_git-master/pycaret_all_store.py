# %% ------------------------------------------------------------------
# LOAD
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from pycaret.regression import *


# %% ------------------------------------------------------------------
# GENERATE FEATURES
df0 = pd.read_pickle('data/df_5.pkl').drop(columns='month_idx')
df = df0.copy().reset_index().set_index(['store', 'date']).sort_index()

# promo: 1 if Jan or Nov (biggest sale), 0 otherwise
df['major_promo'] = np.where((df.month == 1) | (df.month == 11), True, False)

# group by store
df_grouped = df.groupby(level='store')

# lags
df['lag_3'] = df_grouped.shift(3)['sales']

# rolling moving averages
df['mva_2'] = df_grouped.rolling(2).mean()['lag_3'].values
df['mva_3'] = df_grouped.rolling(3).mean()['lag_3'].values
df['mva_4'] = df_grouped.rolling(4).mean()['lag_3'].values
# expanding moving averages
df['ex_mva_2'] = df_grouped.expanding(min_periods=2).mean()['lag_3'].values
df['ex_mva_3'] = df_grouped.expanding(min_periods=3).mean()['lag_3'].values
df['ex_mva_4'] = df_grouped.expanding(min_periods=4).mean()['lag_3'].values
# exponential weighted moving averages
df['ew_mva_2'] = df_grouped.ewm(span=2, min_periods=2).mean()['lag_3'].values
df['ew_mva_3'] = df_grouped.ewm(span=3, min_periods=2).mean()['lag_3'].values
df['ew_mva_4'] = df_grouped.ewm(span=4, min_periods=2).mean()['lag_3'].values

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
         'major_promo',
         'store_area',
         'number_of_staff',
         'holiday_neg',
         'lag_3',
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
df_train = df.reset_index('store').sort_index()[:'2020-09']
df_test = df.reset_index('store').sort_index()['2020-10':'2020-12']
df_unseen = df.reset_index('store').sort_index()['2021-01':]

# check data shape
df_train.shape, df_test.shape, df_unseen.shape


# %% -----------------------------------
# BENCHMARK
X_train = df_train.drop(columns='sales')
y_train = df_train['sales']
X_test = df_test.drop(columns='sales')
y_test = df_test['sales']

# naive (random walk)
fc_naive = np.repeat(y_train[-1:].values, len(y_test))
mae_naive = mean_absolute_error(y_test, fc_naive)
mape_naive = mean_absolute_percentage_error(y_test, fc_naive)

# mean
fc_mean = np.repeat(y_train.mean(), len(y_test))
mae_mean = mean_absolute_error(y_test, fc_mean)
mape_mean = mean_absolute_percentage_error(y_test, fc_mean)

# median
fc_median = np.repeat(y_train.median(), len(y_test))
mae_median = mean_absolute_error(y_test, fc_median)
mape_median = mean_absolute_percentage_error(y_test, fc_median)

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
    'store_level',
    'store_segment',
    'province',
    'year',
    'month',
    'major_promo'
]
numeric_features = [
    'store_area',
    'number_of_staff',
    'holiday_neg',
    'lag_3',
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
    'store_level': ['C', 'B', 'A']
}

ini_setup = setup(data=df_train,
                  target='sales',
                  test_data=df_test,
                  numeric_features=numeric_features,
                  categorical_features=categorical_features,
                  ordinal_features=ordinal_features,
                  
                  # transformation steps
                  normalize=True,
                  pca=True,
                  # interactive features???
                  
                  # CV
                  data_split_shuffle=False,
                  fold_strategy=TimeSeriesSplit(4, test_size=3),
                  
                  use_gpu=True,
                  session_id=3,
                  silent=True
                  )

# %% -----------------------------------------------------------
# spot check all available models
best_model = compare_models(sort='MAPE',
                            turbo=False,  # include all models
                            round=3,
                            n_select=3)  # top 3

# %% -----------------------------------------------------------
# hp tuning

model_1 = best_model[0]
tuned_1 = tune_model(model_1, optimize='MAE')
pred_unseen_1 = predict_model(tuned_1, df_unseen)
MAPE_unseen_1 = mean_absolute_percentage_error(pred_unseen_1['sales'],
                                               pred_unseen_1['Label'])

model_2 = best_model[1]
tuned_2 = tune_model(model_2, optimize='MAE')
pred_unseen_2 = predict_model(tuned_2, df_unseen)
MAPE_unseen_2 = mean_absolute_percentage_error(pred_unseen_2['sales'],
                                               pred_unseen_2['Label'])

model_3 = best_model[2]
tuned_3 = tune_model(model_3, optimize='MAE')
pred_unseen_3 = predict_model(tuned_3, df_unseen)
MAPE_unseen_3 = mean_absolute_percentage_error(pred_unseen_3['sales'],
                                               pred_unseen_3['Label'])

model_rf = RandomForestRegressor()
tuned_rf = tune_model(model_rf, optimize='MAE')
pred_unseen_rf = predict_model(tuned_rf, df_unseen)
MAPE_unseen_rf = mean_absolute_percentage_error(pred_unseen_rf['sales'],
                                                pred_unseen_rf['Label'])

model_xgb = xgb.XGBRegressor()
tuned_xgb = tune_model(model_xgb, optimize='MAE')
pred_unseen_xgb = predict_model(tuned_xgb, df_unseen)
MAPE_unseen_xgb = mean_absolute_percentage_error(pred_unseen_xgb['sales'],
                                                 pred_unseen_xgb['Label'])

# %% summary
all_store_score = pd.DataFrame(
    {'model': [type(model_1).__name__,
               type(model_2).__name__,
               type(model_3).__name__,
               type(model_rf).__name__,
               type(model_xgb).__name__,
               ],
        'MAPE': [round(MAPE_unseen_1, 3),
                 round(MAPE_unseen_2, 3),
                 round(MAPE_unseen_3, 3),
                 round(MAPE_unseen_rf, 3),
                 round(MAPE_unseen_xgb, 3),
                 ],
     }).sort_values('MAPE').reset_index(drop=True)

all_store_score

# %% 
# final_scores_all_store.to_csv('pycaret_all_store.csv')
pd.read_csv('pycaret_all_store.csv').iloc[:,1:]
