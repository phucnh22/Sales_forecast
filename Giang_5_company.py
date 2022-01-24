# %%
# imports
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error, mean_absolute_error, mean_absolute_percentage_error
import pandas as pd
import numpy as np
import plotly.express as px
from pmdarima.preprocessing import FourierFeaturizer
from pmdarima import auto_arima, ARIMA
import matplotlib.pyplot as plt
pd.options.plotting.backend = 'plotly'

#%%##########################################################################
# WHOLE COMPANY
#############################################################################
df_store = pd.read_pickle('data/df_daily.pkl')
train_data = df_store.groupby('date').sum()['sales']/1e6
train_data.index.freq = 'D'

fouri_terms = FourierFeaturizer(365.25, 2)
y_prime, exog = fouri_terms.fit_transform(train_data)
exog.index = y_prime.index


# %%
# Split the time series as well as exogenous features data into train and test splits
steps_ahead = 14
y_to_train = y_prime.iloc[:-steps_ahead]
y_to_test = y_prime.iloc[-steps_ahead:]

exog_to_train = exog.iloc[:-steps_ahead]
exog_to_test = exog.iloc[-steps_ahead:]


# %%
# Fit model to the level to find common order
# Weekly seasonality covered by SARIMAX
arima_exog_model_company = auto_arima(
    y=y_to_train,
    exogenous=exog_to_train,
    D=1,
    seasonal=True,
    m=7)
arima_exog_model_company.summary()

# Forecast
y_arima_exog_fitted = arima_exog_model_company.predict_in_sample(
    X=exog_to_train)
y_arima_exog_forecast = arima_exog_model_company.predict(
    n_periods=len(y_to_test), exogenous=exog_to_test)
y_arima_exog_forecast = pd.Series(y_arima_exog_forecast,
                                  name='forecast',
                                  index=y_to_test.index)


# metrics
# in-sample
mae_IS = round(mean_absolute_error(y_to_train, y_arima_exog_fitted))
mape_IS = round(mean_absolute_percentage_error(
    y_to_train, y_arima_exog_fitted), 3)

# out-sample
mae_OOS = round(mean_absolute_error(y_to_test, y_arima_exog_forecast))
mape_OOS = round(mean_absolute_percentage_error(
    y_to_test, y_arima_exog_forecast), 3)

print(f'Company:')
print('Out of sample:',
      f'MAPE: {mape_OOS}',
      f'MAE: {mae_OOS}\n',
      sep='\n')

res_com = res_com.append({
    'mape_OOS': mape_OOS,
    'mape_IS': mape_IS,
    'mae_OOS': mae_OOS,
    'mae_IS': mae_IS,
    'model': arima_exog_model_company.to_dict(),
    'store_level': 'Company',
    'method': 'common',
}, ignore_index=True)

res_com = res_com.drop(38).reset_index(drop=True)

#%%
print(res_com.loc[38, 'model'])
company_order = (1, 0, 3)
company_S_order = (2, 1, 0, 7)

res_whole = pd.DataFrame()

#%%##########################################################################
# LOOP all stores
for store in df_store['store_id'].unique():#[:2]: print(store)
    train_data = df_store[df_store['store_id'] == store].set_index('date')['sales']
    train_data = train_data/1e6
    train_data.index.freq = 'D'

    # Prepare the fourier terms to model annual seasonality; add as exogenous features to auto_arima
    fouri_terms = FourierFeaturizer(365.25, 2)
    y_prime, exog = fouri_terms.fit_transform(train_data)
    # is exactly the same as manual calculation in the above cells
    exog.index = y_prime.index

    # Split the time series as well as exogenous features data into train and test splits
    steps_ahead = 14
    y_to_train = y_prime.iloc[:-steps_ahead]
    y_to_test = y_prime.iloc[-steps_ahead:]

    exog_to_train = exog.iloc[:-steps_ahead]
    exog_to_test = exog.iloc[-steps_ahead:]

    # Fit model to each store in the level
    arima_exog_model_store = ARIMA(
        order=company_order,
        seasonal_order=company_S_order,
        m=7
    ).fit(y_to_train,
        X=exog_to_train
        )

    # Forecast
    y_arima_exog_fitted = arima_exog_model_store.predict_in_sample(
        X=exog_to_train)
    y_arima_exog_forecast = arima_exog_model_store.predict(
        n_periods=len(y_to_test), exogenous=exog_to_test)
    y_arima_exog_forecast = pd.Series(y_arima_exog_forecast,
                                    name='forecast',
                                    index=y_to_test.index)


    # metrics
    # in-sample
    mae_IS = round(mean_absolute_error(y_to_train, y_arima_exog_fitted))
    mape_IS = round(mean_absolute_percentage_error(
        y_to_train, y_arima_exog_fitted), 3)

    # out-sample
    mae_OOS = round(mean_absolute_error(y_to_test, y_arima_exog_forecast))
    mape_OOS = round(mean_absolute_percentage_error(
        y_to_test, y_arima_exog_forecast), 3)

    print(f'Store {store}')
    print('Out of sample:',
        f'MAPE: {mape_OOS}',
        f'MAE: {mae_OOS}\n',
        sep='\n')

    res_whole = res_whole.append({
        'store_id': store,
        'mape_OOS': mape_OOS,
        'mape_IS': mape_IS,
        'mae_OOS': mae_OOS,
        'mae_IS': mae_IS,
        'model': arima_exog_model_store},
        ignore_index=True)

#%%
# res_whole.to_csv('res_whole.csv')
res_whole = pd.read_csv('res_whole.csv', index_col=0)
