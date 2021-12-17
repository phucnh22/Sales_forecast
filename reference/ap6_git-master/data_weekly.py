# %% ---------------------------------------------------------
# Data loading
import numpy as np
import pandas as pd

# raw data
fact_order = pd.read_pickle('data/fact_order.pkl')
dim_store = pd.read_pickle('data/dim_store.pkl')
df = fact_order.join(dim_store.set_index('store_id'), on='store_id').drop('store_id', axis=1).reset_index()

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
date_start = datetime.date(2018, 1, 1)
# date_start = datetime.date(2017, 9, 1)
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

# add holiday
import holidays
holiday = pd.DataFrame(holidays.Vietnam(years=[2018, 2019, 2020, 2021]).items()).rename({0:'date', 1:'holiday_neg'}, axis=1).set_index('date')
holiday.index = pd.PeriodIndex(holiday.index, freq='D')
# set to 1 if holiday affect sales negatively
holiday.replace({'Vietnamese New Year.*': 1, 
                 '.*day of Tet Holiday': 1,
                 'International Labor Day': 1,
                 '\D': 0}, 
                regex=True, inplace=True)

df_2.index = df_2.index.to_period('D')
df_2 = pd.merge(df_2, holiday, how='left', left_index=True, right_index=True)

# resample to weekly frequency
df_3 = df_2.groupby('store').resample('W').sum().reset_index()

# rejoin with store data
df_4 = pd.merge(df_3.set_index('store'), dim_store.set_index('store_name'), left_index=True, right_index=True)
df_4.index.name = 'store'
df_4 = df_4.set_index('date', append=True)

#%%
df_6 = df_4[['sales','store_group', 'store_format', 'store_level', 'store_segment', 'province', 'store_area', 'number_of_staff', 'holiday_neg']]

df_6.to_pickle('data/df_6.pkl')
# df_5.to_csv('df_5.csv')
