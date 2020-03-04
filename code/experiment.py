import sys
sys.path.insert(0, "../models/")
import gr4h
from anns import constructor

import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution

import keras
from keras.models import load_model


### IMPORTANT ###
# It is highly recommended to use any modern GPU 
# (e.g., NVIDIA 1080Ti, P100, V100, 2060, 2070, 2080)
# for running this script.
# The average time needed to perform the entire set
# of experiments is almost two weeks when using 1080Ti
# or P100 GPUs.

# Probably, running this script on a standard CPU will take forever. 


### uncomment when using GPU ###
#import os
#import tensorflow as tf
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#config = tf.ConfigProto(log_device_placement=True)
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config)


# 1. Reading the data
# uncomment if you requested the data from the authors
#data = pd.read_pickle("../data/data.pkl")

# placeholder for input data
# comment out with # mark if you use the observational data
# which was provided to you by the authors 
data = pd.DataFrame({"Q": np.random.random(324360), 
                     "P": np.random.random(324360), 
                     "PE": np.random.random(324360)
                     }, 
                     index=pd.date_range("1968-01-01 00:00", periods=324360, freq="1H"))


# split data for calibration and validation periods
data_cal = data["1968":"1987"]
data_val = data["1988":"2004"]


# 2. Utils


def mse(y_true, y_pred):
    return np.nanmean((y_true - y_pred) ** 2, axis=0)


def nse(y_true, y_pred):
    return 1 - np.nansum((y_true-y_pred)**2)/np.nansum((y_true-np.nanmean(y_true))**2)


def data_generator(data_instance, model_type, history=720, mode="calibration", runoff_threshold=0.0):
    """
    Data generator function for efficient training of neural networks
    Input:
    data_instance: pandas dataframe with Q, P, and PE columns represent
    discharge, precipitation, and potential evapotranspiration timeseries, respectively
    model_type: one of "GR4H", "MLP", "RNN", "LSTM", "GRU"
    history: the number of antecedent timesteps to consider, hours (default=720, aka one month)
    mode: "calibration" or "validation"
    runoff threshold: the value below which discharge is not considered for calibration, float (default=0.0)
    Output:
    list of variables needed for model calibration / validation
    """

    if model_type == "GR4H":
        
        _Q = data_instance["Q"].values
        _P = data_instance["P"].values
        _PE= data_instance["PE"].values
    
        # add warmup
        # simply add a full period as a warm-up
        Qobs = np.concatenate([_Q,  _Q])
        P    = np.concatenate([_P,  _P])
        PE   = np.concatenate([_PE, _PE])
        
        output = [Qobs, P, PE]
    
    elif model_type in ["RNN", "GRU", "LSTM", "MLP"]:
        
        X_matrix = data_instance[["P", "PE"]].values
        y_matrix = data_instance[["Q"]].values
        
        X, y = [], []
        
        for i in range(history, len(data_instance)):
            
            X_chunk = X_matrix[i-history:i, ::]
            y_chunk = y_matrix[i, ::]
            
            if mode == "training":
                # check for NaNs and non-zero runoff
                if np.isnan(np.sum(X_chunk)) or np.isnan(np.sum(y_chunk)) or y_chunk<runoff_threshold:
                    pass
                else:        
                    X.append(X_chunk)
                    y.append(y_chunk)
            else:
                X.append(X_chunk)
                y.append(y_chunk)
        
        # from lists to np.array
        X, y = np.array(X), np.array(y)
        
        
        # normalization
        X_mean = np.nanmean(X)
        X_std = np.nanstd(X)

        y_mean = np.nanmean(y)
        y_std = np.nanstd(y)

        X -= X_mean
        X /= X_std

        y -= y_mean
        y /= y_std

        
        if model_type == "MLP":
            X = X.reshape(X.shape[0], -1)
        else:
            pass
        
        output = [X, np.squeeze(y), y_mean, y_std]
    
    return output


def calibration(data_instance, model_type, history=720):
    """
    Calibration routine
    Input:
    data_instance: pandas dataframe (the same that for data_generator func)
    model_type: one of "GR4H", "MLP", "RNN", "LSTM", "GRU"
    history: the number of antecedent timesteps to consider, hours (default=720, aka one month)
    Output:
    list of: (1) optimal parameters (or Keras model instance) and pandas dataframe
    with simulation results
    """

    if model_type == "GR4H":
        
        Qobs, P, PE = data_generator(data_instance=data_instance, model_type=model_type)
        
        def loss_gr4h(params):
            # calculate runoff
            Qsim = gr4h.run(P, PE, params)
            # mse on peiod with cropped warm-up
            return np.nanmean((Qobs[-len(data_instance):] - Qsim[-len(data_instance):]) ** 2, axis=0)
        
        # optimization
        opt_par = differential_evolution(loss_gr4h, bounds=gr4h.bounds(), maxiter=100, polish=True, disp=False, seed=42).x
        
        # calculate runoff with optimal parameters
        Qsim = gr4h.run(P, PE, opt_par)
    
        # cut the warmup period + history (for consistency with DL)
        Qobs = Qobs[-len(data_instance)+history:]
        Qsim = Qsim[-len(data_instance)+history:]
        
        print(f"NSE on calibration is {np.round(nse(Qobs, Qsim), 2)}")
	
	# save results from calibration period separately
        calib_res = pd.DataFrame({"Qobs": Qobs, "Qsim": Qsim})
        
        return opt_par, calib_res
    
    elif model_type in ["RNN", "GRU", "LSTM", "MLP"]:
        
        # generate data
        X, y, y_mean, y_std = data_generator(data_instance=data_instance, model_type=model_type)
        
        # create a model instance
        model = constructor(model_type=model_type)
        print(model.summary())
        
        # set callbacks
        # interrupt training if there is no improvement for 100 epochs
        # save the best model on disk
        callbacks_list = [keras.callbacks.EarlyStopping(patience=100), 
                          keras.callbacks.ModelCheckpoint(filepath=f"../models/{model_type}.h5", save_best_only=True)]
        
        # set up training parameters
        validation_split = 0.25
        epochs = 1000
        batch_size=4096
        
        # fit the model
        model.fit(X, y, 
                  validation_split=validation_split,
                  epochs=epochs, 
                  batch_size=batch_size, 
                  callbacks=callbacks_list)
        
        # load the best model
        model = load_model(f"../models/{model_type}.h5")
        
        # prediction
        Qsim = model.predict(X, batch_size=batch_size).reshape(-1)
        # postprocessing
        Qsim *= y_std
        Qsim += y_mean
        Qsim = np.where(Qsim < 0, 0, Qsim)
        
        Qobs = y.copy()
        Qobs *= y_std
        Qobs += y_mean
        
        print(f"NSE on calibration is {np.round(nse(Qobs, Qsim), 2)}")

	# save results from calibration period separately
        calib_res = pd.DataFrame({"Qobs": Qobs, "Qsim": Qsim})
        
        return model, calib_res


def validation(data_instance, model_type, model_instance, history=720):
    """
    Validation routine
    Input:
    data_instance: pandas dataframe (the same that for data_generator func)
    model_type: one of "GR4H", "MLP", "RNN", "LSTM", "GRU"
    history: the number of antecedent timesteps to consider, hours (default=720, aka one month)
    model_instance: optimal parameters of GR4H or Keras model instance of pretrained model
    Output:
    simulated discharge timeseries
    """
    
    if model_type == "GR4H":
        
        # generate data
        Qobs, P, PE = data_generator(data_instance=data_instance, model_type=model_type, history=history, mode="validation")
        
        # calculate runoff with optimal parameters
        Qsim = gr4h.run(P, PE, model_instance)
        
        # cut the warmup period + history (for consistency with DL)
        Qobs = Qobs[-len(data_instance)+history:]
        Qsim = Qsim[-len(data_instance)+history:]
           
    elif model_type in ["RNN", "GRU", "LSTM", "MLP"]:
        
        # generate data
        X, y, y_mean, y_std = data_generator(data_instance=data_instance, model_type=model_type, history=history, mode="validation")
        
        # prediction
        Qsim = model_instance.predict(X, batch_size=1024).reshape(-1)
        # postprocessing
        Qsim *= y_std
        Qsim += y_mean
        Qsim = np.where(Qsim < 0, 0, Qsim)
        
        Qobs = y.copy()
        Qobs *= y_std
        Qobs += y_mean
    
    print(f"NSE on validation is {np.round(nse(Qobs, Qsim), 2)}")
        
    return Qsim


def periods_constructor(duration, year_start, year_end, stride=1):
    """
    Construction of individual calibration periods
    Input:
    duration: the required duration in calender years, int
    year_start: the first year considered, int
    year_end: the last year considered, int
    stride: default=1
    Output:
    the list of considered calender years
    """
    
    duration = duration - 1
    
    periods=[]
    
    while year_end - duration >= year_start:
        
        period = [year_end-duration, year_end]
        
        periods.append(period)
        
        year_end = year_end - stride
    
    return periods


# 3. The main function describes the experiment
# about the evaluation of the effect of calibration data length
# on the performance of different hydrological models


def experiment(calibration_instance, validation_instance, model_type, history=720):
    """
    Input:
    calibtration_instance: pandas dataframe from the respective data generator
    validation_instance: pandas dataframe from the respective data generator
    model_type: one of "GR4H", "MLP", "RNN", "LSTM", "GRU"
    history: the number of antecedent timesteps to consider, hours (default=720, aka one month)
    """

    # loop over different possible caibration period duration
    for period_duration in range(1,21): 
        
        # create an instance of available periods
        periods = periods_constructor(period_duration, calibration_instance.index.year[0], calibration_instance.index.year[-1])

        # initialize a container for storing
        # simulated runoff on validation
        Qsim_container = []
        colnames_container = []
        
        for period in periods:
            
            print(period, model_type)

            # set up years for slicing
            year_start, year_end = period

            # create chunk for calibration
            calibration_chunck = calibration_instance[str(year_start):str(year_end)]

            # calibrate our model
            model, calib_results = calibration(data_instance=calibration_chunck, model_type=model_type, history=history)

	        # save calibration results separately
            calib_results.to_pickle(f"../results/calibration/{model_type}_duration_{period_duration}_period_{year_start}_{year_end}.pkl")

            # run model on validation period
            Qsim = validation(data_instance=validation_instance, model_type=model_type, model_instance=model, history=history)

            # store results in a container
            Qsim_container.append(Qsim)

            # create a respective colname
            colnames_container.append(f"Qsim_{year_start}_{year_end}")

        Qsim_container = np.moveaxis(np.array(Qsim_container), 0, -1)

        results = pd.DataFrame(data=Qsim_container, index=validation_instance.index[history:], columns=colnames_container)
    
        results["Qobs"] = validation_instance["Q"].iloc[history:]
        
        results.to_pickle(f"../results/validation/{model_type}_duration_{period_duration}.pkl")


# 4. The Run

for model_type in ["GR4H", "MLP", "RNN", "LSTM", "GRU"]:
    runoff = experiment(data_cal, data_val, model_type, 720)

