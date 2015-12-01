__author__ = 'prnbs'

import numpy as np
import matplotlib.pyplot as plt
import math


from sklearn.cluster import KMeans
import pandas as pd
from sklearn.cluster import KMeans

traps = pd.read_csv('../input/train_weather_spray_appended.csv', verbose=True, parse_dates=['Date'])

trap_loc = traps[['Longitude', 'Latitude']]
traps = traps.drop('Latitude', 1)
traps = traps.drop('Longitude', 1)

day = []
week_of_year = []
year = []
month = []

for date in traps['Date']:
    week_of_year.append(date.weekofyear)
    year.append(date.year)
    month.append(date.month)
    day.append(date.day)

traps = traps.drop('Date', 1)

clf = KMeans()
labels = clf.fit_predict(trap_loc)

traps['location_cluster'] = labels
traps['WeekOfYear'] = week_of_year
traps['Month'] = month
traps['Year'] = year
traps['Day'] = day
# traps['PrecipTotal_Corr'] = traps['PrecipTotal_Corr'].apply(np.log)

traps.to_csv('../input/train_weather_spray_clustered.csv', index=False)

# print labels




