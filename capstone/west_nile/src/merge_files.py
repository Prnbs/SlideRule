__author__ = 'psinha4'

import pandas as pd
import numpy as np
import math


def distance(lat1, long1, lat2, long2):
    return math.sqrt((lat1-lat2)**2, (long1-long2)**2)


if __name__ == '__main__':
    traps = pd.read_csv('../input/train_clean.csv', parse_dates=['Date'])
    weather = pd.read_csv('../input/weather_corrected.csv', parse_dates=['Date'])
    spray = pd.read_csv('../input/spray.csv', parse_dates=['Date'])

    # new columns to hold info if a location was sprayed and if yes then how many days ago
    # it is initialized to NumMosquitos for no particular reason
    sprayed = np.full((len(traps), 1), -2)
    last_sprayed = np.full((len(traps), 1), -2)
    distance_from_sprayed = np.full((len(traps), 1), -2)

    unique_spray_dates = spray.groupby('Date', as_index=False).count()

    dates_sprayed = unique_spray_dates['Date']

    spray_idx = 0
    traps_idx = 0

    # keep iterating over train until day diff is positive
    while  traps_idx < len(traps):
        day_diff = traps.iloc[traps_idx, 0] - spray.iloc[spray_idx, 0]
        # print day_diff.days, traps.iloc[traps_idx, 0], spray.iloc[spray_idx, 0]
        if day_diff.days < 0:
            sprayed[traps_idx] = False
            last_sprayed[traps_idx] = -1
            traps_idx += 1
        else:
            break

    print "First non negative date at ", traps_idx

    # now update the two new arrays
    # first need to solve closest pair of points problem
    trap_idx_last_item = traps_idx
    l_subdivision_index = []
    for index, row in unique_spray_dates.iterrows():
        # if this is not the last date on which spray was applied
        if index+1 < len(dates_sprayed):
            next_date = dates_sprayed[index+1]
            # find the index up to which this sprayed data can be used
            while trap_idx_last_item < len(traps):
                date = traps.iloc[trap_idx_last_item, 0]
                if date < next_date:
                    sprayed[trap_idx_last_item] = True
                    # print  (date - dates_sprayed[index]).days
                    last_sprayed[trap_idx_last_item] =  (date - dates_sprayed[index]).days
                    trap_idx_last_item += 1
                else:
                    break
            l_subdivision_index.append(trap_idx_last_item)
            print trap_idx_last_item, (date - dates_sprayed[index]).days
        else:
            # last date on spray's list
            next_date = dates_sprayed[index]
            while trap_idx_last_item < len(traps):
                date = traps.iloc[trap_idx_last_item, 0]
                if date > next_date:
                        sprayed[trap_idx_last_item] = True
                        # print  (date - dates_sprayed[index]).days
                        last_sprayed[trap_idx_last_item] = (date - dates_sprayed[index]).days
                        trap_idx_last_item += 1
                else:
                    break

    # sprayed_series = pd.Series(sprayed, index=traps.index)
    # last_sprayed_series = pd.Series(last_sprayed,  index=traps.index)

    # traps['sprayed'] = sprayed
    traps['when_sprayed'] = last_sprayed

    merged = traps.merge(weather, how="inner", on='Date')
    # merged = merged[merged.Station != 2]
    merged.to_csv('../input/train_weather_spray.csv', index=False)

    # traps.to_csv('../input/traps_spray.csv')
