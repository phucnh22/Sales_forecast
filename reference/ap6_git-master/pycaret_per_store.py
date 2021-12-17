# %% -----------------------------------
# LOAD
import numpy as np
import pandas as pd
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from pycaret.regression import *


def eval_MAPE(y_true, y_pred):
    mape = np.abs(y_pred - y_true) / y_true
    return mape.mean()


# %%
df_5 = pd.read_pickle('data/df_5.pkl')
store_name = 'AEON BÌNH TÂN'
df = df_5.loc[df_5['store'] == store_name, ['sales']]
store_list = df_5['store'].unique().tolist()

final_scores_all = []
# %%
for store_name in store_list:
    df = df_5.loc[df_5['store'] == store_name, ['sales']]

    # %% -----------------------------------
    # GENERATE FEATURES
    # lags
    df['lag_3'] = df['sales'].shift(3)
    df['lag_4'] = df['sales'].shift(4)
    df['lag_5'] = df['sales'].shift(5)
    df['lag_6'] = df['sales'].shift(6)

    # moving averages
    df['mva_2'] = df['lag_3'].rolling(2).mean().values
    df['mva_3'] = df['lag_3'].rolling(3).mean().values
    df['mva_4'] = df['lag_3'].rolling(4).mean().values

    df['ex_mva_2'] = df['lag_3'].expanding(min_periods=2).mean().values
    df['ex_mva_3'] = df['lag_3'].expanding(min_periods=3).mean().values
    df['ex_mva_4'] = df['lag_3'].expanding(min_periods=4).mean().values

    df['ew_mva_2'] = df['lag_3'].ewm(span=2, min_periods=2).mean().values
    df['ew_mva_3'] = df['lag_3'].ewm(span=3, min_periods=2).mean().values
    df['ew_mva_4'] = df['lag_3'].ewm(span=4, min_periods=2).mean().values

    # holiday: number of days off in month
    df_holiday = pd.read_pickle('data/holiday.pkl')
    df_holiday.index = df_holiday.index.to_period('M')
    df = pd.merge(df, df_holiday, how='left',
                  left_index=True, right_index=True)

    # promo: 1 if Jan or Nov (biggest sale), 0 otherwise
    df['major_promo'] = np.where(
        (df.index.month == 1) | (df.index.month == 11), True, False)

    # drop NAs
    df = df.dropna().reset_index()

    # convert date to month, year categories
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df = df.drop(columns='date')

    # train/test split
    df_train = df[:-6]
    df_test = df[-6:-3]
    df_unseen = df[-3:]

    # %% -----------------------------------
    # BENCHMARK
    y_train = df_test['sales']
    y_test = df_unseen['sales']
    fc_naive = np.repeat(y_train[-1:].values, len(y_test))
    mape_naive = eval_MAPE(y_test, fc_naive)
    fc_mean = np.repeat(y_train.mean(), len(y_test))
    mape_mean = eval_MAPE(y_test, fc_mean)
    fc_median = np.repeat(y_train.median(), len(y_test))
    mape_median = eval_MAPE(y_test, fc_median)

    # %% -----------------------------------------------------------
    # intialize the setup
    numeric_features = df.loc[:, 'lag_3':'holiday_neg'].columns.values.tolist()
    categorical_features = ['major_promo',
                            'year',
                            'month',]

    ini_setup = setup(data=df_train,
                      target='sales',
                      test_data=df_test,
                      numeric_features=numeric_features,
                      categorical_features=categorical_features,
                      normalize=True,
                      pca=True,
                      data_split_shuffle=False,
                      fold_strategy=TimeSeriesSplit(4, test_size=3),
                      # interactive features???
                      use_gpu=True,
                      session_id=3,
                      silent=True,
                      html=False,
                      )

    # %% -----------------------------------------------------------
    # spot check all available models
    best_model = compare_models(include=['ada',
                                         'rf',
                                         XGBRegressor()],
                                sort='MAPE',
                                n_select=3,
                                round=3)

    # %% -----------------------------------------------------------
    # hp tuning
    def MAPE_tuned(unseen_data, model):
        tuned_model = tune_model(model, optimize='MAE')
        pred_unseen = predict_model(tuned_model, df_unseen)
        MAPE_unseen = eval_MAPE(pred_unseen['sales'], pred_unseen['Label'])
        return tuned_model, MAPE_unseen

    tuned_1, MAPE_unseen_1 = MAPE_tuned(df_unseen, best_model[0])
    tuned_2, MAPE_unseen_2 = MAPE_tuned(df_unseen, best_model[1])
    tuned_3, MAPE_unseen_3 = MAPE_tuned(df_unseen, best_model[2])
    
    # %% summary
    final_scores = pd.DataFrame(
        {'model': [
            type(tuned_1).__name__,
            type(tuned_2).__name__,
            type(tuned_3).__name__,
            '------------------naive',
            '------------------mean',
            '----------------median'],
         'MAPE': [
            round(MAPE_unseen_1, 3),
            round(MAPE_unseen_2, 3),
            round(MAPE_unseen_3, 3),
            round(mape_naive, 3),
            round(mape_mean, 3),
            round(mape_median, 3)],
         'store': store_name,
         }).sort_values('MAPE').reset_index(drop=True)
    final_scores_all.append(final_scores)

# %%
per_store_score = pd.concat([x for x in final_scores_all])
per_store_score.groupby('store').head(1).sort_values('MAPE')

#%%
df_train
df_test
df_unseen
get_config('X')
get_config('X_train')
get_config('X_test')
get_config('y')
get_config('y_train')
get_config('y_test')
get_config('data_before_preprocess')
get_config('fold_generator')
get_config('fold_groups_param')
