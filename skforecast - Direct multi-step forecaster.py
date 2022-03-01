# skforecast - Direct multi-step forecaster
# %%
# Libraries
# ==============================================================================
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import ARIMA
import imp
from skforecast.model_selection_statsmodels import grid_search_sarimax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skforecast.ForecasterAutoregMultiOutput import ForecasterAutoregMultiOutput
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
import warnings
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5
# %matplotlib inline
warnings.filterwarnings('ignore')

# %%
df_store = pd.read_pickle('data/df_daily.pkl')
df_company = df_store.groupby('date').sum()['sales']/1e6
data = df_company.copy()
data.index.freq = 'D'

lags = 7
steps = 3
data_train = data[:-steps]
data_test = data[-steps:]

# %%
# Direct SARIMAX:
#   Option 1: get train/test partition for SARIMAX directly from source code
# ==============================================================================
SARIMAX_model = SARIMAX(endog=data_train, 
                        order=(4, 0, 0),
                        seasonal_order=(2, 1, 0, 7),
                        )
SARIMAX_model = SARIMAX_model.fit()
SARIMAX_model.summary()
SARIMAX_fc_recursive = SARIMAX_model.forecast(3)


# %%
# Direct SARIMAX:
#   Option 2: fit any regressor, get train/test partition for SARIMAX
# ==============================================================================

forecaster = ForecasterAutoregMultiOutput(
    regressor=Ridge(random_state=123),
    lags=lags, steps=steps
)

forecaster.fit(y=data_train)
forecaster.regressors_
forecaster.get_coef(1)

# %%
X, y = forecaster.create_train_X_y(data_train)

# X and y to train model for step 0
X_0, y_0 = forecaster.filter_train_X_y_for_step(
    step=0,
    X_train=X,
    y_train=y,
)

# X and y to train model for step 1
X_1, y_1 = forecaster.filter_train_X_y_for_step(
    step=1,
    X_train=X,
    y_train=y,
)

X_2, y_2 = forecaster.filter_train_X_y_for_step(
    step=2,
    X_train=X,
    y_train=y,
)

data_train  # .tail(10)
X
