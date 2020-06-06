import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_train_test_data(steps=168):
    csv_path = "~/Desktop/Programming/Python/AI/Datasets/OtherData/Temprature/temprature-2004-2013.csv"
    temprature = pd.read_csv(csv_path, index_col=0).values

    train_data = temprature[:-576]
    validation_data = temprature[-576:-288]
    test_data = temprature[-288:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data = scaler.fit_transform(train_data)
    validation_data = scaler.transform(validation_data)
    test_data = scaler.transform(test_data)

    x_train, y_train = split_train_test(steps, train_data)
    x_validation, y_validation = split_train_test(steps, validation_data)
    x_test, y_test = split_train_test(steps, test_data)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_validation = np.array(x_validation)
    y_validation = np.array(y_validation)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return scaler, x_train, y_train, x_validation, y_validation, x_test, y_test

def split_train_test(steps, data):
    x_data = []
    y_data = []

    for day in range(steps, data.shape[0]):
        x_data.append(data[day - steps: day - 24])
        y_data.append(data[day - 24: day][:, 0])

    return x_data, y_data
