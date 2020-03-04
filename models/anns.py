import keras
from keras.models import Sequential
from keras import layers
from keras import optimizers

def constructor(model_type="MLP",
                history=720, 
                hindcast=1,
                input_vars=2, 
                loss="mse", 
                optimizer=optimizers.Adam()):
    
    # model instance initialization
    model = Sequential()
    
    # add a core layer
    if model_type == "MLP":
        model.add(layers.Dense(64, input_shape=(history * input_vars,), activation="relu"))
    elif model_type == "RNN":
        model.add(layers.SimpleRNN(64, return_sequences=False, input_shape=(history, input_vars)))
        optimizer = optimizers.Adam(clipvalue=0.5)
    elif model_type == "GRU":
        model.add(layers.GRU(64, return_sequences=False, input_shape=(history, input_vars)))
    elif model_type == "LSTM":
        model.add(layers.LSTM(64, return_sequences=False, input_shape=(history, input_vars)))
    
    # add the Dense layer on top
    model.add(layers.Dense(hindcast))
    
    # compilation
    model.compile(loss=loss, optimizer=optimizer)

    return model
