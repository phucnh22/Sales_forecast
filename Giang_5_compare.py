# %%
# imports
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sqlalchemy import column


# %%
res_diff = pd.read_csv('res_diff.csv', index_col=0)
res_com = pd.read_csv('res_com.csv', index_col=0)
res_whole = pd.read_csv('res_whole.csv', index_col=0)
res_diff.sort_values('store_level', inplace=True)
res_com.sort_values('store_level', inplace=True)

res_com['method'] = 'level'
res_diff['method'] = 'indiv'
res_whole['method'] = 'whole'
res_whole['store_level'] = 'whole'

res = pd.concat([res_whole, res_com, res_diff]).drop(columns='model').sort_values(['store_id', 'method'])


#%% compare scores of 2 methods
res_mae = res.loc[res['method']=='indiv', 'mae_IS'] - res.loc[res['method']=='different', 'mae_IS']
px.line(res_mae)
res_mape = res.loc[res['method']=='common', 'mape_IS'] - res.loc[res['method']=='different', 'mape_IS']
px.line(res_mape)

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


#%% 
res_summary = round(res.groupby('method').mean().sort_index(), 2)
res_summary.drop(columns='store_id', inplace=True)

res_summary[['mae_IS','mae_OOS']].plot()
res_summary[['mape_IS','mape_OOS']].plot()

px.scatter(res_summary[['mae_IS','mae_OOS']])
px.scatter(res_summary[['mape_IS','mape_OOS']])
