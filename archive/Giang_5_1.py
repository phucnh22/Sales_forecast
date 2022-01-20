# %%
# imports
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error, mean_absolute_error, mean_absolute_percentage_error
import pandas as pd
import numpy as np
import plotly.express as px
from pmdarima.preprocessing import FourierFeaturizer
from pmdarima import auto_arima, ARIMA
import matplotlib.pyplot as plt

# %%
# DATA
df_store = pd.read_pickle('data/df_daily.pkl')
df_store = df_store[['date', 'sales', 'store_level', 'store_id']]
store_list = df_store.store_id.unique()

train_data = df_store[df_store['store_id'] == store_list[0]].set_index('date')[
    'sales']
train_data = train_data[:'2021-01-31']/1e6


# %%
# Prepare the fourier terms to model annual seasonality; add as exogenous features to auto_arima
fouri_terms = FourierFeaturizer(365.25, 2)
y_prime, exog = fouri_terms.fit_transform(train_data)
# is exactly the same as manual calculation in the above cells
exog.index = y_prime.index


# %%
# Split the time series as well as exogenous features data into train and test splits
steps_ahead = 14
y_to_train = y_prime.iloc[:-steps_ahead]
y_to_test = y_prime.iloc[-steps_ahead:]

exog_to_train = exog.iloc[:-steps_ahead]
exog_to_test = exog.iloc[-steps_ahead:]


# %%
# Fit model
# Weekly seasonality covered by SARIMAX
arima_exog_model_store = ARIMA(
    order=arima_exog_model.order,
    seasonal_order=arima_exog_model.seasonal_order,
    m=7
).fit(y_to_train,
      X=exog_to_train
      )
arima_exog_model_store.summary()


# %%
# Forecast
y_arima_exog_fitted = arima_exog_model_store.predict_in_sample(X=exog_to_train)
y_arima_exog_forecast = arima_exog_model_store.predict(
    n_periods=len(y_to_test), exogenous=exog_to_test)
y_arima_exog_forecast = pd.Series(y_arima_exog_forecast,
                                  name='forecast',
                                  index=y_to_test.index)


# %%
# metrics
# in-sample
mae = round(mean_absolute_error(y_to_train, y_arima_exog_fitted))
mape = round(mean_absolute_percentage_error(
    y_to_train, y_arima_exog_fitted), 3)
print('Train score:',
      f'MAPE: {mape}',
      f'MAE: {mae}\n', sep='\n')

# out-sample
mae = round(mean_absolute_error(y_to_test, y_arima_exog_forecast))
mape = round(mean_absolute_percentage_error(
    y_to_test, y_arima_exog_forecast), 3)
print('Test score:',
      f'MAPE: {mape}',
      f'MAE: {mae}', sep='\n')


# %%
# Plots
# in-sample
dat_IS = pd.DataFrame({'train': y_to_train,
                       'fitted': y_arima_exog_fitted},
                      index=y_to_train.index).reset_index()

fig = px.line(dat_IS, x='date', y=['train', 'fitted'])
fig.show()

# %% out-of-sample
dat_OOS = pd.DataFrame({'test': y_to_test,
                        'forecast': y_arima_exog_forecast},
                       index=y_to_test.index).reset_index()
fig = px.line(dat_OOS, x='date', y=['test', 'forecast'])
fig.show()


#%%
