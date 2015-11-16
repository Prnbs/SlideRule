__author__ = 'prnbs'


import pandas as pd
import numpy as np
import math


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def correct_station_data(row_index, col_index, weather):
    result = 0
    try:
        if not is_number(weather.iloc[row_index, col_index]):
            if not is_number(weather.iloc[row_index+1, col_index]):
                result = 0
            else:
                result = weather.iloc[row_index+1, col_index]
        else:
            if is_number(weather.iloc[row_index+1, col_index]):
                result = max(float(weather.iloc[row_index, col_index]), float(weather.iloc[row_index+1, col_index]))
            else:
                result = float(weather.iloc[row_index, col_index])
    except IndexError:
        print row_index
    return result


if __name__ == '__main__':

    weather = pd.read_csv('../input/weather.csv', parse_dates=['Date'])
    station_idx = weather.columns.get_loc("Station")

    precip_corrected = np.full((len(weather), 1), 0)
    wetBulb_corrected = np.full((len(weather), 1), 0)
    stnPres_corrected = np.full((len(weather), 1), 0)
    seaLevel_corrected = np.full((len(weather), 1), 0)

    precip_idx = weather.columns.get_loc("PrecipTotal")
    wetbulb_idx = weather.columns.get_loc("WetBulb")
    stnPres_idx = weather.columns.get_loc("StnPressure")
    sealevel_idx = weather.columns.get_loc("SeaLevel")

    for index, row in weather.iterrows():
        # ignore the even rows
        if weather.iloc[index, station_idx] == 2:
            continue
        if index >= len(weather):
            break
        precip_corrected[index] = correct_station_data(index, precip_idx, weather)
        wetBulb_corrected[index] = correct_station_data(index, wetbulb_idx, weather)
        seaLevel_corrected[index] = correct_station_data(index, sealevel_idx, weather)
        stnPres_corrected[index] = correct_station_data(index, stnPres_idx, weather)

    weather['PrecipTotal_Corr'] = precip_corrected
    weather['WetBulb_Corr'] = wetBulb_corrected
    weather['SeaLevel_Corr'] = seaLevel_corrected
    weather['StnPress_Corr'] = stnPres_corrected

    weather = weather.drop("PrecipTotal", 1)
    weather = weather.drop("WetBulb", 1)
    weather = weather.drop("StnPressure", 1)
    weather = weather.drop("SeaLevel", 1)
    weather = weather.drop("CodeSum", 1)
    weather = weather.drop("Depth", 1)
    weather = weather.drop("Water1", 1)
    weather = weather.drop("SnowFall", 1)

    weather = weather[weather.Station != 2]
    weather = weather.drop("Station", 1)

    weather.to_csv('../input/weather_corrected.csv', index=False)

