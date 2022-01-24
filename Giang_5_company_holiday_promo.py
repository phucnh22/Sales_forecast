# %%
# imports
import imp
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
# adding holidays & promo as df_fouri. variables
#############################################################################
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
# Weekly seasonality covered by SARIMAX
arima_model = auto_arima(
    y=y_to_train,
    exogenous=exog_to_train,
    D=1,
    seasonal=True,
    m=7)
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
for store in df_store['store_id'].unique()[26:]:# print(store)
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

    # metrics
    # in-sample
    mae_IS = round(mean_absolute_error(y_to_train, arima_y_fitted))
    mape_IS = round(mean_absolute_percentage_error(
        y_to_train, arima_y_fitted), 3)

    # out-sample
    mae_OOS = round(mean_absolute_error(y_to_test, arima_y_forecast))
    mape_OOS = round(mean_absolute_percentage_error(
        y_to_test, arima_y_forecast), 3)

    print(f'Store {store}')
    print(f'MAPE: {mape_OOS}',
          f'MAE: {mae_OOS}\n',
          sep='\n')

    res_whole_holiday_promo = res_whole_holiday_promo.append({
        'store_id': store,
        'mape_OOS': mape_OOS,
        'mape_IS': mape_IS,
        'mae_OOS': mae_OOS,
        'mae_IS': mae_IS
        }, ignore_index=True)

# %%
# res_whole.to_csv('res_whole_holiday_promo.csv')
res_whole_holiday_promo = pd.read_csv('res_whole_holiday_promo.csv', index_col=0)
res_whole = pd.read_csv('res_whole.csv', index_col=0)

res_whole_holiday_promo.mean()
res_whole.mean()
