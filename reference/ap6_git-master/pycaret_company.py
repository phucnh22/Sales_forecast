# %% -----------------------------------------------------------
# import
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from pycaret.regression import *

# %% -----------------------------------------------------------
# intialize the setup
numeric_features = df.loc[:, 'lag_3':'holiday_neg'].columns.values.tolist()
categorical_features = ['major_promo',
                        'month',
                        'year']

ini_setup = setup(data=df_train,
                  target='sales',
                  test_data=df_test,
                  numeric_features=numeric_features,
                  categorical_features=categorical_features,
                #   normalize=True,
                #   pca=True,
                  data_split_shuffle=False,
                  fold_strategy=TimeSeriesSplit(n_splits=4, test_size=3),
                  # interactive features???
                  use_gpu=True,
                  session_id=3,
                  silent=True
                  )


# %% -----------------------------------------------------------
# spot check all available models
best_model = compare_models(sort='MAPE',
                            turbo=False,  # include all models
                            round=3,
                            n_select=3)  # top 3
# best_model

# %% -----------------------------------------------------------
# hp tuning

def MAPE_tuned(unseen_data, model):
    tuned_model = tune_model(model, optimize='MAE')
    pred_unseen = predict_model(tuned_model, df_unseen)
    MAPE_unseen = mean_absolute_percentage_error(pred_unseen['sales'],
                                                 pred_unseen['Label'])
    return tuned_model, MAPE_unseen


tuned_1, MAPE_unseen_1 = MAPE_tuned(df_unseen, best_model[0])
tuned_2, MAPE_unseen_2 = MAPE_tuned(df_unseen, best_model[1])
tuned_3, MAPE_unseen_3 = MAPE_tuned(df_unseen, best_model[2])
tuned_rf, MAPE_unseen_rf = MAPE_tuned(df_unseen, RandomForestRegressor())
tuned_xgb, MAPE_unseen_xgb = MAPE_tuned(df_unseen, xgb.XGBRegressor())

# summary
final_scores_company = pd.DataFrame(
    {'model': [type(tuned_1).__name__,
               type(tuned_2).__name__,
               type(tuned_3).__name__,
               type(tuned_rf).__name__,
               type(tuned_xgb).__name__],
     'MAPE': [round(MAPE_unseen_1, 3),
              round(MAPE_unseen_2, 3),
              round(MAPE_unseen_3, 3),
              round(MAPE_unseen_rf, 3),
              round(MAPE_unseen_xgb, 3)]
     }).sort_values('MAPE').reset_index(drop=True)
# final_scores_company.to_csv('pycaret_company.csv')

final_scores_company

#%% combine
tuned_models = [tuned_1, tuned_2, tuned_3, tuned_rf, tuned_xgb]
preds = []
for m in tuned_models:
    pred_unseen = predict_model(m, df_unseen)['Label']
    preds.append(pred_unseen)
preds = pd.DataFrame(preds).T

mean_absolute_percentage_error(df_unseen['sales'], preds.mean(axis=1))

#%%
pd.read_csv('pycaret_company.csv').iloc[:,1:]
