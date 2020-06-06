from create_datasets import create_train_test_data
import matplotlib.pyplot as plt
from keras import models
import numpy as np

scaler, x_train, y_train, x_validation, y_validation, x_test, y_test = create_train_test_data()

days = range(168)

model = models.load_model("./temprature_lstm_1.h5")

predict_past_data = x_train[0]
predicted_temprature = model.predict(predict_past_data.reshape(1, 144, 4))

past_temprature = predict_past_data[:, 0]
feature_temprature = predicted_temprature[0]
predicted_data = np.concatenate([past_temprature, feature_temprature])
plt.plot(days, predicted_data, c="r")

real_data = np.concatenate([past_temprature, y_train[0]])
plt.plot(days, real_data, c="b")

plt.show()
