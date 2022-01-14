# %%
# compare results by store_levels
# daily
# forecast horizon: 14 days
# Naive, ARIMA, Prophet
# fc strategy: package default (comment in code)

import pandas as pd


# %% data
df = pd.read_pickle('data/df_daily.pkl')[['date', 'store_id', 'sales', 'store_level']].dropna()
df = df.replace({
    'A++': 'A',
    'A+': 'B',
    'A': 'C',
    'B': 'D',
    'C': 'E'})
# number of stores in each level
df.groupby('store_level').nunique()['store_id']

df_wide = df.set_index('date').sort_values(['store_level', 'store_id']).pivot(columns=['store_level', 'store_id'], values='sales')
ts = df_wide.iloc[:,2]/1e6
ts.name = ts.name[1]

#%% 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
sns.set_style("darkgrid")
plt.rc("figure", figsize=(16, 12))
plt.rc("font", size=15)

from statsmodels.tsa.seasonal import STL
#%%
stl = STL(ts['2018'], 
          robust=True)
res = stl.fit()
fig = res.plot()

#%%
stl.period
