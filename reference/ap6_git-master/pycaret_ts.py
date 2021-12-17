#%%
import time
import numpy as np
import pandas as pd
from pycaret.internal.pycaret_experiment import TimeSeriesExperiment


#%%---------------------------------------
ts = pd.read_pickle('data/df_6.pkl')

#%% setup
fh = 14 # or alternately fh = np.arange(1,13)
fold = 4
exp = TimeSeriesExperiment()
exp.setup(data=ts, 
          fh=fh,
          fold=fold,
          session_id=3,
          system_log=False,
          html=False,
          use_gpu=True)

#%%---------------------------------------
# EDA

# time series plot
exp.plot_model()

# ACF and PACF for the original dataset
exp.plot_model(plot="acf")
exp.plot_model(plot="pacf", data_kwargs={'nlags':18})

# decom
exp.plot_model(plot="decomp_classical")
exp.plot_model(plot="decomp_classical", data_kwargs={'type': 'multiplicative'})
exp.plot_model(plot="decomp_stl")

# Show the train-test splits on the dataset
exp.plot_model(plot="train_test_split")

# Show the Cross Validation splits inside the train set
exp.plot_model(plot="cv")

# Plot diagnostics
exp.plot_model(plot="diagnostics")

# Tests
exp.check_stats()


#%%---------------------------------------
# MODELING

# baseline models
best_baseline_models = exp.compare_models(sort='mape', n_select=3, turbo=False)
best_baseline_models

compare_metrics = exp.pull()
compare_metrics

# tune models
best_tuned_models = [exp.tune_model(model) for model in best_baseline_models]

exp.predict_model(best_tuned_models[0])
exp.predict_model(best_tuned_models[1])
exp.predict_model(best_tuned_models[2])

#%%---------------------------------------
# TRADITIONAL MODELS
model_naive = exp.create_model('naive')
exp.tune_model(model_naive)
