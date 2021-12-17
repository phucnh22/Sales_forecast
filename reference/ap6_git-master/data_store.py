# %% ---------------------------------------------------------
# Data loading
import numpy as np
import pandas as pd

# raw data
fact_order = pd.read_pickle('data/fact_order.pkl')
dim_store = pd.read_pickle('data/dim_store.pkl')
df = fact_order.join(dim_store.set_index('store_id'), on='store_id').drop('store_id', axis=1)

# %% ---------------------------------------------------------
# Data cleaning

# rename columns
df.columns = ['date', 'sales', 'store', 'store_group', 'store_format', 'store_level', 'store_segment', 'store_area', 'number_of_staff', 'province']

# set datetime index
df.date = pd.to_datetime(df.date)
df.set_index('date', inplace=True)

# remove wholesale orders (sales >20mil)
df = df.loc[df.sales < 20000000]

# trim dates
import datetime
# date_start = datetime.date(2018, 1, 1)
date_start = datetime.date(2017, 9, 1)
date_end = datetime.date(2021, 3, 31)
df = df.sort_index().loc[date_start:date_end+datetime.timedelta(days=1)]

# keep only stores with data from date_start to date_end; remove the rest
store_groups = df.groupby('store')
df = store_groups.filter(lambda group: (group.index.min().date() == date_start) & (group.index.max().date() == date_end))
df.store.drop_duplicates()#.shape

# missing values
df_1 = df[['sales', 'store']].groupby('store').resample('D').sum()
df_1.replace(0, np.nan, inplace=True)
df_1.sort_values('sales')
nan_idx = np.where(df_1.sales.isna())
df_1.iloc[nan_idx]
df_2 = (df_1.groupby('store')).transform(lambda x : x.interpolate()).reset_index('store')
df_2.iloc[nan_idx]

# resample to monthly frequency
df_3 = df_2.groupby('store').resample('M').sum().reset_index()

# rejoin with store data
df_store = df.reset_index().drop(['date', 'sales'], axis=1).drop_duplicates()
df_4 = pd.merge(df_3, df_store, how='left', on='store').set_index(['store', 'date'])
df_4.groupby('store').size()

# add holiday
df_4 = df_4.reset_index('store')
df_4.index = df_4.index.to_period('M')
df_4

# 
import holidays
holiday = pd.DataFrame(holidays.Vietnam(years=[2018, 2019, 2020, 2021]).items()).rename({0:'date', 1:'holiday_neg'}, axis=1).set_index('date')
holiday.index = pd.PeriodIndex(holiday.index, freq='D')

# set to 1 if holiday affect sales negatively
holiday.replace('Vietnamese New Year.*', 1, regex=True, inplace=True)
holiday.replace('.*day of Tet Holiday', 1, regex=True, inplace=True)
holiday.replace('International Labor Day', 1, regex=True, inplace=True)
holiday.replace('\D', 0, regex=True, inplace=True)
holiday = holiday.resample('M').sum()

#%%
df_5 = pd.merge(df_4, holiday, how='left', left_index=True, right_index=True)
# df_5.reset_index('store', inplace=True)
df_5['month'] = df_5.index.month.astype('object')
df_5['year'] = df_5.index.year.astype('object')
df_5['month_idx'] = (df_5['month'] + (df_5['year'] -df_5['year'].min())*12).astype('int')

df_5 = df_5[['sales', 'store', 'store_group', 'store_format', 'store_level', 'store_segment', 'province', 'year', 'month', 'month_idx', 'store_area', 'number_of_staff', 'holiday_neg']]

df_5.to_pickle('data/df_5.pkl')
# df_5.to_csv('df_5.csv')
