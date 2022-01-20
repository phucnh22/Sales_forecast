# %%
# imports
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt


# %%
res_diff = pd.read_csv('res_diff.csv', index_col=0)
res_com = pd.read_csv('res_com.csv', index_col=0)
res_diff.sort_values('store_level', inplace=True)
res_com.sort_values('store_level', inplace=True)

res_diff.groupby('store_level').mean()
res_com.groupby('store_level').mean()

res_com['method'] = 'common'
res_diff['method'] = 'different'

res = pd.concat([res_diff, res_com]).drop(columns='model').sort_values('store_id')

#%% compare scores of 2 methods

#%%
fig = px.line(res, x='store_id', y='mape_IS', color='method')
fig.update_xaxes(type='category')
fig.show()

#%%
fig = px.line(res, x='store_id', y='mape_OOS', color='method')
fig.update_xaxes(type='category')
fig.show()

#%%
fig = px.line(res, x='store_id', y='mae_IS', color='method')
fig.update_xaxes(type='category')
fig.show()

#%%
fig = px.line(res, x='store_id', y='mae_OOS', color='method')
fig.update_xaxes(type='category')
fig.show()


#%% investigate params for each store
res = pd.concat([res_diff, res_com]).set_index(['method','store_id']).sort_index()
store
res.loc[('common', store),'model']

print(res['model'][0])
a = res['model'][0]
l = str(a).split('\n')
l[23]
