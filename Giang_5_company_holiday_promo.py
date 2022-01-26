# %%
# imports
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
import itertools
from prophet import Prophet
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error, mean_absolute_error, mean_absolute_percentage_error
import pandas as pd
import numpy as np
import plotly.express as px
from pmdarima.preprocessing import FourierFeaturizer
from pmdarima import auto_arima, ARIMA
import matplotlib.pyplot as plt
import holidays
pd.options.plotting.backend = 'plotly'


#%%##########################################################################
# WHOLE COMPANY
# with holidays & promo as exog. variables

df_store = pd.read_pickle('data/df_daily.pkl')
df_company = df_store.groupby('date').sum()
train_data = df_company['sales']/1e6
train_data.index.freq = 'D'

# yearly seasonality
fouri_terms = FourierFeaturizer(365.25, 2)
y_prime, df_fouri = fouri_terms.fit_transform(train_data)
df_fouri.index = y_prime.index

# holiday
ts_holiday = pd.read_pickle('data/holiday.pkl')

# promo
ts_promo = df_company['promo_count']

# combine exog. variables
df_exog = pd.concat([df_fouri, ts_holiday, ts_promo], axis=1)
df_exog['holiday'] = df_exog['holiday'].fillna(False).astype('int')


# %%
# Split the time series as well as exogenous features data into train and test splits
steps_ahead = 14
y_to_train = y_prime.iloc[:-steps_ahead]
y_to_test = y_prime.iloc[-steps_ahead:]

exog_to_train = df_exog.iloc[:-steps_ahead]
exog_to_test = df_exog.iloc[-steps_ahead:]


# %%
# Fit model to the level to find common order
arima_model = auto_arima(
    y=y_to_train,
    exogenous=exog_to_train,
    D=1, 
    seasonal=True, m=7 # Weekly seasonality
    )
arima_model

# %%
# Forecast
arima_y_fitted = arima_model.predict_in_sample(
    X=exog_to_train)
arima_y_forecast = arima_model.predict(
    n_periods=len(y_to_test), exogenous=exog_to_test)
arima_y_forecast = pd.Series(arima_y_forecast,
                             name='forecast',
                             index=y_to_test.index)


# %%
# metrics
# in-sample
mae_IS = round(mean_absolute_error(y_to_train, arima_y_fitted))
mape_IS = round(mean_absolute_percentage_error(
    y_to_train, arima_y_fitted), 3)

# out-sample
mae_OOS = round(mean_absolute_error(y_to_test, arima_y_forecast))
mape_OOS = round(mean_absolute_percentage_error(
    y_to_test, arima_y_forecast), 3)

print(f'Company:')
print('Out of sample:',
      f'MAPE: {mape_OOS}',
      f'MAE: {mae_OOS}\n',
      sep='\n')

# %%
print(arima_model)
res_whole_holiday_promo = pd.DataFrame()

#%%##########################################################################
# LOOP all stores
for store in df_store['store_id'].unique():  # print(store)
    df_data = df_store[df_store['store_id'] == store].set_index('date')[
        ['sales', 'promo_count']]
    df_data.index.freq = 'D'
    train_data = df_data['sales']/1e6

    # Prepare the fourier terms to model annual seasonality; add as exogenous features to auto_arima
    fouri_terms = FourierFeaturizer(365.25, 2)
    y_prime, df_fouri = fouri_terms.fit_transform(train_data)
    # is exactly the same as manual calculation in the above cells
    df_fouri.index = y_prime.index

    # holiday
    # ts_holiday = pd.read_pickle('data/holiday.pkl')

    # promo
    ts_promo = df_data['promo_count']

    # combine exog. variables
    df_exog = pd.concat([df_fouri, ts_holiday, ts_promo], axis=1)
    df_exog['holiday'] = df_exog['holiday'].fillna(False).astype('int')
    df_exog.dropna(inplace=True)

    # Split the time series as well as exogenous features data into train and test splits
    steps_ahead = 14
    y_to_train = y_prime.iloc[:-steps_ahead]
    y_to_test = y_prime.iloc[-steps_ahead:]

    exog_to_train = df_exog.iloc[:-steps_ahead]
    exog_to_test = df_exog.iloc[-steps_ahead:]

    # Fit model to each store in the level
    arima_model_store = ARIMA(
        order=arima_model.order,
        seasonal_order=arima_model.seasonal_order
    ).fit(y_to_train,
          X=exog_to_train
          )

    # Forecast
    arima_y_fitted = arima_model_store.predict_in_sample(
        X=exog_to_train)
    arima_y_forecast = arima_model_store.predict(
        n_periods=len(y_to_test), exogenous=exog_to_test)
    arima_y_forecast = pd.Series(arima_y_forecast,
                                 name='forecast',
                                 index=y_to_test.index)
    # Random walk
    RW_y_forecast = pd.Series(
        y_to_train[-1], name='RW_forecast', index=y_to_test.index)

    # metrics
    # in-sample
    mae_IS = round(mean_absolute_error(y_to_train, arima_y_fitted))
    mape_IS = round(mean_absolute_percentage_error(
        y_to_train, arima_y_fitted), 3)

    # out-sample
    mae_OOS = round(mean_absolute_error(y_to_test, arima_y_forecast))
    mape_OOS = round(mean_absolute_percentage_error(
        y_to_test, arima_y_forecast), 3)
    RW_mae_OOS = round(mean_absolute_error(y_to_test, RW_y_forecast))
    RW_mape_OOS = round(mean_absolute_percentage_error(
        y_to_test, RW_y_forecast), 3)

    print(f'Store {store}')
    print(f'MAPE: {mape_OOS}',
          f'MAE: {mae_OOS}\n',
          sep='\n')

    res_whole_holiday_promo = res_whole_holiday_promo.append({
        'store_id': store,
        'fc_IS': arima_y_fitted,
        'fc_OOS': arima_y_forecast,
        'fc_RW': RW_y_forecast,
        'arima_mape_OOS': mape_OOS,
        'arima_mape_IS': mape_IS,
        'arima_mae_OOS': mae_OOS,
        'arima_mae_IS': mae_IS,
        'RW_mape_OOS': RW_mape_OOS,
        'RW_mae_OOS': RW_mae_OOS,
    }, ignore_index=True)
# res_whole_holiday_promo.to_csv('res_whole_holiday_promo.csv')
# res_whole_holiday_promo = pd.read_csv('res_whole_holiday_promo.csv', index_col=0)


# %%

px.scatter(fb_df, 'promo_count', 'sales')
px.scatter(fb_df, 'holiday', 'sales')
fb_df[['sales', 'promo_count']].plot()


#%%##########################################################################
# PROPHET
#############################################################################

fb_df = pd.concat([df_company[['sales', 'promo_count']], ts_holiday], axis=1)
fb_df['sales'] = fb_df['sales']/1e6
fb_df['holiday'] = fb_df['holiday'].fillna(False).astype('bool')
fb_df = fb_df.reset_index().rename({'date': 'ds', 'sales': 'y'}, axis=1)
fb_train = fb_df.iloc[:-steps_ahead]
fb_test = fb_df.iloc[-steps_ahead:]

param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
}

all_params = [dict(zip(param_grid.keys(), v))
              for v in itertools.product(*param_grid.values())]

# %%
mape_PROPHET = []
for params in all_params:
    # set up model
    m = Prophet(**params)
    m.add_regressor('holiday')
    m.add_regressor('promo_count')
    m.fit(fb_train)
    # set up CV
    df_cv = cross_validation(
        m,
        initial=(str(fb_train.shape[0]-steps_ahead*4)+' days'),
        period='7 days', horizon='14 days')
    # evaluate
    df_p = performance_metrics(df_cv)
    mape_PROPHET.append(df_p['mape'].values[-1])

# Find the best parameters
tuning_results = pd.DataFrame(all_params)
tuning_results['mape'] = mape_PROPHET
tuning_results.sort_values('mape', inplace=True)

# %% final PROPHET
fb_model = Prophet(
    changepoint_prior_scale=tuning_results.loc[0, 'changepoint_prior_scale'],
    seasonality_prior_scale=tuning_results.loc[0, 'seasonality_prior_scale'],
)

fb_model.fit(fb_df)
fb_fc = fb_model.predict(fb_test)
mape_PROPHET = round(mape(y_test_fb['y'], fb_fc['yhat']), 3)
mape_PROPHET

# %%
res_fb = pd.DataFrame()

#%%##########################################################################
# LOOP all stores
for store in df_store['store_id'].unique():  # print(store)
    df_data = df_store[df_store['store_id'] == store].set_index('date')[
        ['sales', 'promo_count']]
    fb_df = pd.concat([df_data, ts_holiday], axis=1)
    fb_df['sales'] = fb_df['sales']/1e6
    fb_df['holiday'] = fb_df['holiday'].fillna(False).astype('bool')
    fb_df = fb_df.dropna().reset_index().rename(
        {'date': 'ds', 'sales': 'y'}, axis=1)
    fb_train = fb_df.iloc[:-steps_ahead]
    fb_test = fb_df.iloc[-steps_ahead:]

    fb_model = Prophet(
        changepoint_prior_scale=tuning_results.loc[0,'changepoint_prior_scale'],
        seasonality_prior_scale=tuning_results.loc[0,'seasonality_prior_scale'],
    )
    fb_model.add_regressor('holiday')
    fb_model.add_regressor('promo_count')
    fb_model.fit(fb_train)
    fb_fc = fb_model.predict(fb_test)

    # metrics
    fb_mae_OOS = round(mean_absolute_error(fb_test['y'], fb_fc['yhat']))
    fb_mape_OOS = round(mean_absolute_percentage_error(
        fb_test['y'], fb_fc['yhat']), 3)

    print(f'Store {store}')
    print(f'MAPE: {fb_mape_OOS}',
          f'MAE: {fb_mae_OOS}\n',
          sep='\n')

    res_fb = res_fb.append({
        'store_id': store,
        'fb_fc': fb_fc['yhat'],
        'fb_mape_OOS': fb_mape_OOS,
        'fb_mae_OOS': fb_mae_OOS,
    }, ignore_index=True)

# res_fb.to_csv('res_fb.csv')
# res_fb = pd.read_csv('res_fb.csv')

# %%
res = pd.merge(res_fb.set_index('store_id')['fb_mape_OOS'],
               res_whole_holiday_promo.set_index(
                   'store_id')[['RW_mape_OOS', 'arima_mape_OOS']],
               'inner', left_index=True, right_index=True).reset_index()


# %%
fig = res[['fb_mape_OOS',
        #    'RW_mape_OOS',
           'arima_mape_OOS'
           ]].plot()

# RW_mean = round(res['RW_mape_OOS'].mean(), 3)
# fig.add_hline(y=RW_mean, line_dash="dot", line_color='red',
#               annotation_text=str(RW_mean))

arima_mean = round(res['arima_mape_OOS'].mean(), 3)
fig.add_hline(y=arima_mean, line_dash="dot", line_color='green',
              annotation_text=str(arima_mean))

fb_mean = round(res['fb_mape_OOS'].mean(), 3)
fig.add_hline(y=fb_mean, line_dash="dot", line_color='blue',
              annotation_text=str(fb_mean),
              annotation_position="bottom right")
