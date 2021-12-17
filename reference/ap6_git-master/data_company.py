# %% --------------------------------
# LOAD
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

def mean_absolute_percentage_error(y_true, y_pred):
    mape = np.abs(y_pred - y_true) / y_true
    return mape.mean()

df_actual_revenue = pd.read_pickle('data/fact_order.pkl')[['actual_revenue']]


# %% --------------------------------
# CLEAN
df = df_actual_revenue.copy()

# rename columns
df.columns = ['sales']
df.index.name = 'date'

# remove wholesale orders (sales >20mil)
df = df.loc[df.sales < 20000000]

# resample to daily
df = df.resample('D').sum()

# trim dates
df = df['2017-09-01':'2021-03-31']

# resample to monthly
df_sum = df.resample('M').sum()
df_mean = df.resample('M').mean()
df = df_sum

# %% --------------------------------
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
df = pd.merge(df, df_holiday, how='left', left_index=True, right_index=True)

# promo: 1 if Jan or Nov (biggest sale), 0 otherwise
df['major_promo'] = np.where((df.index.to_period('M').month == 1) | (df.index.to_period('M').month == 11), True, False)

# drop NAs
df = df.dropna().reset_index()

# convert date to month, year categories
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df = df.drop(columns='date')

# train/test split
df_train = df[:-3]
df_test = df[-6:-3]
df_unseen = df[-3:]

# check data shape
df_train.shape, df_test.shape, df_unseen.shape

# %% --------------------------------
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

#%%
print('''Benchmark MAPE:\n
Naive:  {:.3}
Mean:   {:.3}
Median: {:.3}'''
    .format(mape_naive, mape_mean, mape_median))


#%%
