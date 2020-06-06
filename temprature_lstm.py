from keras.models import Sequential
from keras import layers

def define_lstm(train_shape):
    model = Sequential()

    model.add(layers.LSTM(64, return_sequences = True, input_shape = (train_shape[1], train_shape[2])))
    model.add(layers.Dropout(0.2))

    model.add(layers.LSTM(64))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(64))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(24))

    model.compile(optimizer = "adam", loss = "mean_squared_error")

    print(model.summary())

    return model
