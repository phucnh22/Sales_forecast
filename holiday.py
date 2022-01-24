# %%
# holiday
import holidays
holiday = pd.DataFrame(holidays.Vietnam(years=[2017, 2018, 2019, 2020, 2021]).items()).rename({0: 'date', 1: 'holiday'}, axis=1).set_index('date')
holiday.index = pd.DatetimeIndex(holiday.index)

# set to 1 if holiday affect sales negatively
off_holiday = ['Vietnamese New Year.*',
               '.*day of Tet Holiday',
               'International Labor Day']
holiday.replace(off_holiday, 1, regex=True, inplace=True)
holiday = holiday.loc[holiday['holiday'] == 1, 'holiday'].astype('bool')
holiday = holiday.loc['2017-08':'2021-01']

#%%
holiday.to_pickle('data/holiday.pkl')
