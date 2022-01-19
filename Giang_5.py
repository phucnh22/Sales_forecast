# %%
# imports
import pandas as pd
import numpy as np
import plotly.express as px
from pmdarima.preprocessing import FourierFeaturizer
from pmdarima import auto_arima
import matplotlib.pyplot as plt

# %%
# DATA
df_lev = pd.read_pickle('data/df_lev.pkl')
level = 'A'
train_data = df_lev[df_lev['store_level'] == level].set_index('date')['sales']
train_data = train_data[:'2021-01-31']


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
arima_exog_model = auto_arima(
    y=y_to_train,
    exogenous=exog_to_train,
    D=1,
    seasonal=True,
    m=7)
arima_exog_model.summary()

# Forecast
y_arima_exog_fitted = arima_exog_model.predict_in_sample(X=exog_to_train)
y_arima_exog_forecast = arima_exog_model.predict(
    n_periods=len(y_to_test), exogenous=exog_to_test)
y_arima_exog_forecast = pd.Series(y_arima_exog_forecast,
                                  name='forecast',
                                  index=y_to_test.index)


# %%
# Plots
# %% in-sample
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


# %%
# alternative FC
alt_train_data = df_lev[df_lev['store_level'] == level].set_index('date')[
    'sales']
alt_fouri_terms = FourierFeaturizer(365.25, 2)
alt_y_prime, alt_exog = fouri_terms.fit_transform(alt_train_data)
alt_exog.index = alt_y_prime.index

alt_first_day = '2021-01-18'
alt_y_to_test = alt_y_prime.loc[alt_first_day:]
alt_exog_to_test = alt_exog.loc[alt_first_day:]

alt_forecast = arima_exog_model.predict(
    n_periods=len(alt_y_to_test), exogenous=alt_exog_to_test)
alt_forecast = pd.Series(alt_forecast,
                         name='forecast',
                         index=alt_y_to_test.index)

alt_dat_OOS = pd.DataFrame({'test': alt_y_to_test,
                        'forecast': alt_forecast},
                       index=alt_y_to_test.index).reset_index()
fig = px.line(alt_dat_OOS, x='date', y=['test', 'forecast'])
fig.show()
