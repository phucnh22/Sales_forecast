#%%
import numpy as np
import pandas as pd
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.transformations.series.impute import Imputer

#%%
df_lev = pd.read_pickle('data/df_lev.pkl').set_index('date')
ts = df_lev[df_lev['store_level'] == 'C']['sales']
ts = round(ts/1e6, 3)
ts.index.freq = 'D'
ts.plot()

#%%
steps_ahead = 14
y_train = ts[:-steps_ahead]#.interpolate() # in-sample/train set
y_train.isna().sum()
y_test = ts[-steps_ahead:]  # out-of-sample/test set
k = 4 # number of CV folds
initial_window = len(y_train) - steps_ahead*k
fh = list(range(1, steps_ahead+1))
sp = 7 # seasonal period

#%%
cv = ExpandingWindowSplitter(
    initial_window=initial_window, 
    fh=fh, 
    step_length=steps_ahead)

pipe = TransformedTargetForecaster(steps=[
    ("imputer", Imputer()),
    ("forecaster", NaiveForecaster())
    ])

gscv = ForecastingGridSearchCV(
    forecaster=pipe,
    param_grid=[{
        "forecaster": [NaiveForecaster(sp=1)],
        "forecaster__strategy": ["mean"],
    },
    {
        # "imputer__method": ["linear"],
        "forecaster": [ARIMA(sp=sp)],
        "forecaster__order": param_grid_ARIMA,
        "forecaster__seasonal_order": param_grid_ARIMA_seas,
    },
    ],
    cv=cv,
    n_jobs=-1)

gscv.fit(y_train)
gscv.best_forecaster_

#%%
y_pred = gscv.predict(fh=fh)

