__author__ = 'prnbs'

import numpy as np
import matplotlib.pyplot as plt


from sklearn.cluster import KMeans
import pandas as pd
from sklearn.cluster import KMeans

traps = pd.read_csv('../input/train_weather_spray.csv', verbose=True, parse_dates=['Date'])

trap_loc = traps[['Longitude', 'Latitude']]
traps = traps.drop('Latitude', 1)
traps = traps.drop('Longitude', 1)

date_ordinal = []

for date in traps['Date']:
    date_ordinal.append(date.toordinal())

traps = traps.drop('Date', 1)

clf = KMeans()
labels = clf.fit_predict(trap_loc)

traps['location_cluster'] = labels
traps['Date'] = date_ordinal

traps.to_csv('../input/train_weather_spray.csv', index=False)

print labels




