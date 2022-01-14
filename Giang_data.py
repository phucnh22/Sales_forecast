# %% --------------------------------
# import
import pandas as pd

# transactional data
fact_order = pd.read_pickle('data/fact_order.pkl')
fact_order.head().T
cols = [
    'order_id',
    'actual_revenue',
    'store_id',
    'is_promotion',
    'created_at', ]

fact_order_cleaned = (fact_order
                      .loc[fact_order['actual_revenue'] <= 20000000, cols]
                      .set_index('order_id')
                      .dropna())
fact_order_cleaned = fact_order_cleaned.astype({'store_id': 'object',
                                                'is_promotion': 'bool',
                                                'created_at': 'datetime64'})

fact_order_cleaned = fact_order_cleaned.loc[fact_order_cleaned['created_at'].between(
    pd.Timestamp('2017-08-01'), pd.Timestamp('2021-02-01')), :].sort_values('created_at')

fact_order_cleaned.info()
fact_order_cleaned.head()
fact_order.columns
fact_order_cleaned.set_index('created_at', inplace=True)

# %% 
# resample to daily freq
gr = fact_order_cleaned.groupby('store_id')

fact_order_daily_sum = (gr.resample('D').sum()
                        .reset_index(level=0)
                        .rename({'actual_revenue': 'sales',
                                 'is_promotion': 'promo_count'}, axis=1)
                        )

# first & last days of each store
fact_order_daily_sum['store_id'].unique().shape
gr = fact_order_daily_sum.groupby('store_id')
# for key, val in gr:
#     print(key)
#     print(val.first('D').index.to_period('D').astype('object')[0])
#     print(val.last('D').index.to_period('D').astype('object')[0])
#     print('\n')

# filter: keep stores whose last active day is 2021-01-31 or later
fact_order_daily_sum = fact_order_daily_sum.groupby('store_id').filter(lambda x: (x.last('D').index >= pd.Timestamp('2021-01-31'))[0])

# filter: keep stores with at least 2 full years worth of data
fact_order_daily_sum = fact_order_daily_sum.groupby('store_id').filter(lambda x: (x.first('D').index < pd.Timestamp('2019-01-31'))[0])

fact_order_daily_sum['store_id'] = fact_order_daily_sum['store_id'].astype('int').astype('object')
fact_order_daily_sum['store_id'].value_counts()
store_list = fact_order_daily_sum['store_id'].unique()

fact_order_daily_sum.index.name='date'
fact_order_daily_sum.to_pickle('data/fact_order_daily_sum.pkl')


#%% 
# store info

# read data
dim_store = pd.read_csv('data/dim_store.csv')

# filter data: 
# keep ['Retail', 'Online']
dim_store_cleaned = dim_store[dim_store['channel'].isin(['Retail', 'Online'])]

# keep relevant columns
dim_store_cleaned[dim_store_cleaned['store_id']==307222].T
cols = ['store_id', 
        'store_level',
        'store_group', 
        'store_format', 
        'store_segment', 
        'opening_date', 
        'closed_date', 
        'status', 
        'store_area', 
        'number_of_staff',
        'province', 
        'channel']
dim_store_cleaned = dim_store_cleaned[cols]

# keep stores which do not have closed date 
dim_store_cleaned = dim_store_cleaned[dim_store_cleaned['closed_date'].isna()]
del dim_store_cleaned['closed_date']

# keep stores which appear in fact_order_daily_sum
dim_store_cleaned = dim_store_cleaned[dim_store_cleaned['store_id'].isin(store_list)].reset_index(drop=True)
dim_store_cleaned.to_pickle('data/dim_store_cleaned.pkl')

#%% join dim_store to fact_order
dim_store_cleaned['store_id'] = dim_store_cleaned['store_id'].astype('object')
fact_order_daily_sum.join(dim_store_cleaned, how='inner', on='store_id', lsuffix='_left')

df_daily = pd.merge(fact_order_daily_sum.reset_index(), dim_store_cleaned, 'inner', 'store_id')
df_daily.to_pickle('data/df_daily.pkl')
