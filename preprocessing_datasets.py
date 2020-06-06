import pandas as pd
import os
import numpy as np

datasets_dir = "/Users/katsuyamashouki/Desktop/Programming/Python/AI/Datasets/OtherData/Temprature/"
data_columns = ["temprature", "precipitation", "sunshine", "wind"]

def get_all_data():
    all_data = []
    year_dir = os.path.join(datasets_dir, "Years")
    for year in sorted(os.listdir(year_dir)):
        if year == ".DS_Store":
            continue

        year_path = os.path.join(year_dir, year)
        year_data = []

        for index, season in enumerate(sorted(os.listdir(year_path), key=lambda season: int(season[-6:-4]))):
            if season == ".DS_Store":
                continue

            season_path = os.path.join(year_path, season)
            season_data = preprocessing_season_data(season_path)
            year_data.append(season_data)

        year_data = concat_season_data(year_data)
        all_data.append(year_data)
    all_data = concat_year_data(all_data)
    save_all_data(all_data)

def preprocessing_season_data(season_path):
    season_data = pd.read_table(season_path, encoding="shift-jis", skiprows=range(5), delimiter=",")
    season_index = [date[:-6] for date in season_data.iloc[:, 0]]
    season_data.index = season_index
    season_data = season_data.iloc[:, [1, 4, 8, 12]]
    season_data.columns = data_columns
    season_data = season_data.fillna(0)
    return season_data

def concat_season_data(year_data):
    year = pd.DataFrame(columns=data_columns)

    for season_data in year_data:
        year = pd.concat([year, season_data], axis=0)

    return year

def concat_year_data(all_data):
    data = pd.DataFrame(columns=data_columns)

    for year_data in all_data:
        data = pd.concat([data, year_data], axis=0)

    return data

def save_all_data(all_data):

    all_data.to_csv(os.path.join(datasets_dir, "temprature-2004-2013.csv"))

get_all_data()
