from create_datasets import create_train_test_data
from temprature_lstm import define_lstm
from plot_history import plot

scaler, x_train, y_train, x_validation, y_validation, x_test, y_test = create_train_test_data()

print(y_train.shape)
model = define_lstm(x_train.shape)
model.save("./temprature_lstm_1.h5")
history = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test))
plot(history)
