__author__ = 'prnbs'

import pandas as pd
import numpy as np
import math

def haversine_distance(lat1, lon1):
    lat2 = 41.8369
    lon2 = -87.6847
    R = 3961
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (math.sin(dlat/2))**2 + math.cos(lat1) * math.cos(lat2) * (math.sin(dlon/2))**2
    c = 2 * math.atan2( math.sqrt(a), math.sqrt(1-a) )
    d = R * c
    return c
    # return math.sqrt(dlon**2 + dlat**2)

if __name__ == '__main__':

    traps = pd.read_csv('../input/train.csv', parse_dates=['Date'])
    d_categorical_species = {}
    d_categorical_species['CULEX ERRATICUS'] = 0
    d_categorical_species['CULEX PIPIENS'] = 1
    d_categorical_species['CULEX PIPIENS/RESTUANS'] = 2
    d_categorical_species['CULEX RESTUANS'] = 3
    d_categorical_species['CULEX SALINARIUS'] = 4
    d_categorical_species['CULEX TARSALIS'] = 5
    d_categorical_species['CULEX TERRITANS'] = 6

    species_categorical = np.full((len(traps), 1), 0)
    distance_from_centre = np.full((len(traps), 1), 0)

    species_idx = traps.columns.get_loc("Species")
    lat_idx = traps.columns.get_loc("Latitude")
    long_idx = traps.columns.get_loc("Longitude")

    for index, row in traps.iterrows():
        if d_categorical_species[traps.iloc[index, species_idx]] == 1 or \
            d_categorical_species[traps.iloc[index, species_idx]] == 3:
            species_categorical[index] = 1
        elif d_categorical_species[traps.iloc[index, species_idx]] == 2:
            species_categorical[index] = 1.5
        else:
            species_categorical[index] = 0

        # distance_from_centre = haversine_distance(traps.iloc[index, lat_idx], traps.iloc[index, long_idx])

    traps['IsPipiens'        ] = ((traps['Species']=='CULEX PIPIENS'  )*1 +           # 8.9%   / 2699
                               (traps['Species']=='CULEX PIPIENS/RESTUANS')*0.5)
    traps['IsPipiensRestuans'] = ((traps['Species']=='CULEX PIPIENS/RESTUANS')*1 +    # 5.5%   / 4752
                               (traps['Species']=='CULEX PIPIENS'  )*0 + (traps['Species']=='CULEX RESTUANS'  )*0)
    traps['IsRestuans'       ] = ((traps['Species']=='CULEX RESTUANS'  )*1 +          # 1.8%   / 2740
                               (traps['Species']=='CULEX PIPIENS/RESTUANS')*0.5)
#         traps['IsSalinarius'     ] = (traps['Species']=='CULEX SALINARIUS')*1           #   0%   /   86
#         traps['IsTarsalis'       ] = (traps['Species']=='CULEX TARSALIS'  )*1           #   0%   /    6
#         traps['IsTerritans'      ] = (traps['Species']=='CULEX TERRITANS' )*1           #   0%   /  222
    traps['IsOther'          ] = (traps['Species']!='CULEX PIPIENS')*(traps['Species']!='CULEX PIPIENS/RESTUANS')*(traps['Species']!='CULEX RESTUANS')



    traps = traps.drop("Species", 1)

    # traps['Species_Categorical'] = species_categorical

    traps.to_csv('../input/train_clean.csv', index=False)
