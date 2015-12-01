__author__ = 'psinha4'
import pandas as pd
import csv
import datetime

if __name__ == '__main__':
    traps_actual = pd.read_csv('../input/train_weather_spray.csv', parse_dates=['Date'])
    # only want last week's data for when virus is present
    traps = traps_actual[traps_actual.WnvPresent == 1]
    print "Length before ", len(traps_actual)

    # create a dictionary from weather file with date as key
    reader = pd.read_csv(open("../input/weather_corrected.csv"))
    weather = {}
    for index, row in reader.iterrows():
        # print row[0]
        weather[row[0]] = row[1:]

    for index, row in traps.iterrows():
        # From the day the virus was found add the weather for each day before that date to the dataframe.
        # go back for 5 days
        number_of_days_in_past = 3
        # store the data from traps csv. We'll append it to the weather for the last N days
        data_from_traps = row[1:10]
        # now subtract N days
        for day in range(1, number_of_days_in_past+1):
            data_to_prepend = data_from_traps
            delta = datetime.timedelta(days=day)
            earlier_date = row[0] - delta
            earlier_date_str = earlier_date.strftime("%Y-%m-%d")
            # get the weather for the earlier date
            weather_data_for_earlier_day = weather[earlier_date_str]

            data_to_prepend['Date'] = earlier_date
            # unknown bug causes date with 1 day difference to appear as elapsed seconds, so add it twice
            data_to_prepend['Date'] = earlier_date

            data_to_append = data_to_prepend.append(weather_data_for_earlier_day)
            traps_actual = traps_actual.append(data_to_append, ignore_index=True)

    print "Length after ", len(traps_actual)
    traps_actual.to_csv('../input/train_weather_spray_appended.csv', index=False)

