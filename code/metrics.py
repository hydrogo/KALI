import pandas as pd
import numpy as np
from collections import OrderedDict


### IMPORTANT ###
# This script calculates and aggregates evaluation metrics for the results of 
# the calibration/validation experiment that aims to investigate the effect of
# calibration data length on the performance of different hydrological models.
# To this aim, the script requires two sub-directories exist in 'results' directory:
# namely, 'calibration' and 'validation' that hold raw results of the above experiment.
# There are two ways to have these repositories:
# 1. Download the available results from the Zenodo data repository at 
# https://doi.org/10.5281/zenodo.3696832 and extract the .zip archive in 'results' directory.
# 2. Create two empty directories (i.e., 'calibration' and 'validation') and 
# run the script 'experiment.py' that will calculate everything from scratch.
#
# You do not need to run this scripts if you only want to investigate the final results closer.
# The aggregated evaluation metrics are available in summary_calibration.npy and summary_calibration.npy
# files in 'results' directory.


# Evaluation metrics

def NSE(obs, sim):
	"""
	Nash-Sutcliffe efficiency (or coefficient of determination)
	Input:
	obs: streamflow observations (numpy array or compatible)
	sim: streamflow simulations (numpy array or compatible)
	Output:
	NSE (float)
	"""
    return 1 - np.nansum((obs - sim)**2)/np.nansum((obs - np.nanmean(obs))**2)


def KGE(obs, sim):
	"""
	Kling-Gupta efficiency
	Input:
	obs: streamflow observations (numpy array or compatible)
	sim: streamflow simulations (numpy array or compatible)
	Output:
	KGE (float)
	"""
    mask_nan = np.isnan(obs) | np.isnan(sim)
    r = np.corrcoef(obs[~mask_nan], sim[~mask_nan])[0,1]
    beta  = np.nanmean(sim) / np.nanmean(obs)
    alfa  = np.nanstd(sim) / np.nanstd(obs)
    kge = 1 - np.sqrt( (r-1)**2 + (beta-1)**2 + (alfa-1)**2 )
    return kge #, np.abs(r-1), np.abs(alfa-1), np.abs(beta-1)


def Bias(obs, sim):
	"""
	Percentage Bias
	Input:
	obs: streamflow observations (numpy array or compatible)
	sim: streamflow simulations (numpy array or compatible)
	Output:
	bias (float)
	"""
    return 100 * ( np.nansum(sim - obs) / np.nansum(obs) )


def MSE(obs, sim):
	"""
	Mean squared error
	Input:
	obs: streamflow observations (numpy array or compatible)
	sim: streamflow simulations (numpy array or compatible)
	Output:
	MSE (float)
	"""
    return np.nanmean((obs - sim)**2)


# Placeholders for the results

results_cal = OrderedDict()
results_val = OrderedDict()


for model_name in ["GR4H", "MLP", "RNN", "GRU", "LSTM"]:
       
    results_cal[model_name] = OrderedDict()
    results_val[model_name] = OrderedDict()
    
    for duration in range(1,21):
        data = pd.read_pickle(f"../results/validation/{model_name}_duration_{duration}.pkl")
        members = data.columns.tolist()
        members.remove("Qobs")
        
        results_cal[model_name][duration] = OrderedDict()
        results_val[model_name][duration] = OrderedDict()
        
        for fname, function in zip(["MSE", "NSE", "KGE", "Bias"], [MSE, NSE, KGE, Bias]):
            
            results_cal[model_name][duration][fname] = OrderedDict()
            results_val[model_name][duration][fname] = OrderedDict()
            
            for member in members:
                # validation results
                results_val[model_name][duration][fname][member] = function(data["Qobs"].values, data[member].values)
                
                # calibration results
                data_cal = pd.read_pickle(f"../results/calibration/{model_name}_duration_{duration}_period_{member[5:]}.pkl")
                results_cal[model_name][duration][fname][member] = function(data_cal["Qobs"].values, data_cal["Qsim"].values)


# Save results
np.save("../results/summary_calibration.npy", results_cal)
np.save("../results/summary_validation.npy", results_val)