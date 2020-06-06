from keras import models
from create_datasets import create_train_test_data

def predict_temprature(x_test, y_test):
    model = models.load_model("./temprature_lstm_0.h5")
    score = model.evaluate(x_test, y_test)
    print(score)

x_train, y_train, x_validation, y_validation, x_test, y_test = create_train_test_data()
