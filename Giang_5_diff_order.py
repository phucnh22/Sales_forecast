# %%-
# imports
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error, mean_absolute_error, mean_absolute_percentage_error
import pandas as pd
import numpy as np
import plotly.express as px
from pmdarima.preprocessing import FourierFeaturizer
from pmdarima import auto_arima, ARIMA
import matplotlib.pyplot as plt

# %%-
# DATA
df_store = pd.read_pickle('data/df_daily.pkl')
df_store = df_store[['date', 'sales', 'store_level', 'store_id']]
store_list = df_store.store_id.unique()
# res_diff = pd.DataFrame(columns=['store',
#                                  'mape_OOS',
#                                  'mape_IS',
#                                  'mae_OOS',
#                                  'mae_IS',
#                                  'model',
#                                  ])

# %%
for store in store_list:  # print(store)
    train_data = df_store[df_store['store_id'] == store].set_index('date')[
        'sales']
    train_data = train_data[:'2021-01-31']/1e6

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

    # Fit model
    arima_exog_model_store_order = auto_arima(
        y=y_to_train,
        exogenous=exog_to_train,  # Annual seasonality
        seasonal=True, m=7)  # Weekly seasonality

    # Forecast
    y_arima_exog_fitted = arima_exog_model_store_order.predict_in_sample(
        X=exog_to_train)
    y_arima_exog_forecast = arima_exog_model_store_order.predict(
        n_periods=len(y_to_test), exogenous=exog_to_test)
    y_arima_exog_forecast = pd.Series(y_arima_exog_forecast,
                                      name='forecast',
                                      index=y_to_test.index)

    # in-sample metrics
    mae_IS = round(mean_absolute_error(y_to_train, y_arima_exog_fitted))
    mape_IS = round(mean_absolute_percentage_error(
        y_to_train, y_arima_exog_fitted), 3)

    # out-sample metrics
    mae_OOS = round(mean_absolute_error(y_to_test, y_arima_exog_forecast))
    mape_OOS = round(mean_absolute_percentage_error(
        y_to_test, y_arima_exog_forecast), 3)

    print(store)
    print('Out of sample:',
          f'MAPE: {mape_OOS}',
          f'MAE: {mae_OOS}\n',
          sep='\n')

    res_diff = res_diff.append({
        'store': store,
        'mape_OOS': mape_OOS,
        'mape_IS': mape_IS,
        'mae_OOS': mae_OOS,
        'mae_IS': mae_IS,
        'model': arima_exog_model_store_order.to_dict()},
        ignore_index=True)

## time elapsed: ~ 2 hours




# %%
# res_diff.to_csv('res_diff.csv')
res_diff = pd.read_csv('res_diff.csv', index_col=0)
res_diff.groupby('store_level').mean()
