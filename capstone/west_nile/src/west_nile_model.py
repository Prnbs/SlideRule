__author__ = 'psinha4'

import pandas as pd
import numpy as np
import datetime
from sklearn.cluster import KMeans
from matplotlib.font_manager import FontProperties
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
import os.path


def file_exists(file_name):
    return os.path.exists(file_name)


def actual_file_name(file_name, day):
    index = len(file_name) - 4
    actual_name = file_name[:index] + str(day) + ".csv"
    return actual_name


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


class Model:

    # Cleans train and test files
    # Removes species, address, block, street, address accuracy
    # Breaks down the Species into a one hot like encoder
    def clean_train(self, in_file, day, out):
        out_file = actual_file_name(out, day)
        if file_exists(out_file):
            return

        traps = pd.read_csv(in_file, parse_dates=['Date'])

        traps['Type_Pipiens'] = ((traps['Species'] == 'CULEX PIPIENS')*1 + (traps['Species'] == 'CULEX PIPIENS/RESTUANS')*1)

        traps['Type_Pipiens_Restuans'] = ((traps['Species'] == 'CULEX PIPIENS/RESTUANS')*1)
        traps['Type_Restuans'] = ((traps['Species'] == 'CULEX RESTUANS')*1 + (traps['Species'] == 'CULEX PIPIENS/RESTUANS') * 1)

        traps['Type_Other'] = (traps['Species'] != 'CULEX PIPIENS')*(traps['Species'] != 'CULEX PIPIENS/RESTUANS')*\
                              (traps['Species'] != 'CULEX RESTUANS')*1

        traps = traps.drop("Species", 1)
        traps = traps.drop("Address", 1)
        traps = traps.drop("Block", 1)
        traps = traps.drop("Street", 1)
        traps = traps.drop("AddressNumberAndStreet", 1)
        traps = traps.drop("AddressAccuracy", 1)

        traps.to_csv(out_file, index=False)

    # I use station 1's data
    # Fills up missing data in weather for station 1 with data from the station 2
    # if both stations have data missing assumes zero
    def correct_station_data(self, row_index, col_index, weather):
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

    # Calls correct_station_data() for Precipitation, Wetbulb, Sealevel and StnPressure
    # Deletes codesum, depth, water1 and Snowfall
    def clean_weather(self, in_file, day, out):
        out_file = actual_file_name(out, day)
        if file_exists(out_file):
            return

        weather = pd.read_csv(in_file, parse_dates=['Date'])
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
            precip_corrected[index] = self.correct_station_data(index, precip_idx, weather)
            wetBulb_corrected[index] = self.correct_station_data(index, wetbulb_idx, weather)
            seaLevel_corrected[index] = self.correct_station_data(index, sealevel_idx, weather)
            stnPres_corrected[index] = self.correct_station_data(index, stnPres_idx, weather)

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

        weather.to_csv(out_file, index=False)

    # Merges train or test with weather and spray
    # Spray data only exists from August 2011, so for earlier dates -1 is entered
    # For others this column contains how many days ago was this area last sprayed
    def merge_files(self, in_train_file, in_weather_file, day, out):
        in_train_file = actual_file_name(in_train_file, day)
        in_weather_file = actual_file_name(in_weather_file, day)
        out_file = actual_file_name(out, day)
        if file_exists(out_file):
            return

        traps = pd.read_csv(in_train_file, parse_dates=['Date'])
        weather = pd.read_csv(in_weather_file, parse_dates=['Date'])
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

        spray_date_idx = spray.columns.get_loc("Date")
        traps_date_idx = traps.columns.get_loc("Date")

        # keep iterating over train until day diff is positive
        while traps_idx < len(traps):
            day_diff = traps.iloc[traps_idx, traps_date_idx] - spray.iloc[spray_idx, spray_date_idx]
            # print day_diff.days, traps.iloc[traps_idx, 0], spray.iloc[spray_idx, 0]
            if day_diff.days < 0:
                sprayed[traps_idx] = False
                # last_sprayed[traps_idx] = 100
                last_sprayed[traps_idx] = -1
                traps_idx += 1
            else:
                break

        # print "First non negative date at ", traps_idx

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
                    date = traps.iloc[trap_idx_last_item, traps_date_idx]
                    if date < next_date:
                        sprayed[trap_idx_last_item] = True
                        day_diff = (date - dates_sprayed[index]).days
                        last_sprayed[trap_idx_last_item] = self.sprayed_since_ordinal(day_diff)
                        # last_sprayed[trap_idx_last_item] = day_diff
                        trap_idx_last_item += 1
                    else:
                        break
                l_subdivision_index.append(trap_idx_last_item)
                # print trap_idx_last_item, (date - dates_sprayed[index]).days
            else:
                # last date on spray's list
                next_date = dates_sprayed[index]
                while trap_idx_last_item < len(traps):
                    date = traps.iloc[trap_idx_last_item, traps_date_idx]
                    if date > next_date:
                            sprayed[trap_idx_last_item] = True
                            # print  (date - dates_sprayed[index]).days
                            day_diff = (date - dates_sprayed[index]).days
                            last_sprayed[trap_idx_last_item] = self.sprayed_since_ordinal(day_diff)
                            # last_sprayed[trap_idx_last_item] = day_diff
                            trap_idx_last_item += 1
                    else:
                        break

        # sprayed_series = pd.Series(sprayed, index=traps.index)
        # last_sprayed_series = pd.Series(last_sprayed,  index=traps.index)

        # traps['sprayed'] = sprayed
        traps['when_sprayed'] = last_sprayed

        merged = traps.merge(weather, how="inner", on='Date')
        # merged = merged[merged.Station != 2]
        merged.to_csv(out_file, index=False)

    # When a row declares that virus is present, append N=days_past number of day's weather before
    # the day the virus was found
    def last_week_weather(self, days_past, in_merged_file, in_weather_file, out_file):
        in_merged_file = actual_file_name(in_merged_file, days_past)
        in_weather_file = actual_file_name(in_weather_file, days_past)
        out_file = actual_file_name(out_file, days_past)
        if file_exists(out_file):
            return

        traps_actual = pd.read_csv(in_merged_file, parse_dates=['Date'])
        # only want last week's data for when virus is present
        traps = traps_actual[traps_actual.WnvPresent == 1]
        print "Length before ", len(traps_actual)

        # create a dictionary from weather file with date as key
        reader = pd.read_csv(open(in_weather_file))
        weather = {}
        for index, row in reader.iterrows():
            # print row[0]
            weather[row[0]] = row[1:]

        for index, row in traps.iterrows():
            # From the day the virus was found add the weather for each day before that date to the dataframe.
            # go back for 5 days
            # store the data from traps csv. We'll append it to the weather for the last N days
            data_from_traps = row[1:11]
            # now subtract N days
            for day in range(1, days_past+1):
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
        traps_actual.to_csv(out_file, index=False)

    # Use KMeans to cluster the latitude and longitude data
    def cluster_locations(self, in_file, days_before, test=False, clf=None, out=None):
        in_file = actual_file_name(in_file, days_before)
        out_file = actual_file_name(out, days_before)
        if file_exists(out_file):
            return

        traps = pd.read_csv(in_file, parse_dates=['Date'])

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

        if clf is None:
            clf = KMeans()
            labels = clf.fit_predict(trap_loc)
        else:
            labels = clf.predict(trap_loc)

        traps['location_cluster'] = labels
        traps['WeekOfYear'] = week_of_year
        traps['Month'] = month
        traps['Year'] = year
        traps['Day'] = day
        if not test:
            traps = traps.drop('NumMosquitos', 1)
        traps.to_csv(out_file, index=False)
        return clf

    # Clusters can be names as T001 or T001A where the A means its an extra trap added at the same location
    # This function extracts the number from the trap data
    def cluster_traps(self, in_file, day, clf=None, out=None):
        in_file = actual_file_name(in_file, day)
        out_file = actual_file_name(out, day)
        if file_exists(out_file):
            return
        traps = pd.read_csv(in_file)
        traps.Trap = traps.Trap.map(lambda x: float(x[1:4]))
        traps.to_csv(out_file, index=False)

    # Trains the classifier using stratified K folds and then plots an roc curve
    def train_plot_roc_auc(self, clf, plt, traps, labels, name):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        # Compute micro-average ROC curve and ROC area
        kf = StratifiedKFold(labels, n_folds=10, shuffle=True, random_state =0)

        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        mean_accuracy = 0

        for train_indices, test_indices in kf:
            features_train = traps.iloc[train_indices]
            features_test  = traps.iloc[test_indices]
            labels_train   = [labels[ii] for ii in train_indices]
            labels_test    = [labels[ii] for ii in test_indices]

            clf.fit(features_train, labels_train)
            pred = clf.predict(features_test)

            fpr, tpr, _ = roc_curve(labels_test, pred)
            from scipy import interp
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0

            from sklearn.metrics import accuracy_score
            mean_accuracy += accuracy_score(labels_test, pred)

        print name, mean_accuracy / len(kf)
        label = 'ROC for ' + name +' (area = %0.2f)'
        mean_tpr /= len(kf)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr,
                 label=label % mean_auc)
        return plt, clf

    # Function to group the number of days since last sprayed into specific ordinals
    def sprayed_since_ordinal(self, days):
        # if days <= 5:
        #     return 1
        # elif days <= 10:
        #     return 2
        # elif days <= 15:
        #     return 3
        # elif days <= 20:
        #     return 4
        # elif days <= 30:
        #     return 5
        # elif days <= 60:
        #     return 6
        # elif days <= 90:
        #     return 7
        # elif days <= 120:
        #     return 8
        # else:
        #     return 9
        return days

    # Trains the classifier and saves the roc plot to file
    def fit_model(self, file_num, in_file, plt=None):
        in_file = actual_file_name(in_file, day)
        traps = pd.read_csv(in_file)
        # extract label
        labels = traps['WnvPresent']
        # drop label from features
        traps.drop('WnvPresent', 1, inplace=True)

        from sklearn.ensemble import RandomForestClassifier
        import matplotlib.pyplot as plt
        if plt is None:
            plt.clf()

        # for estimator in [2,10,30,50,100,200]:
        #     print estimator
        clf_rf = RandomForestClassifier(n_estimators=800, min_samples_leaf=74, max_depth=15, criterion='entropy', random_state=0)

        plt, clf = self.train_plot_roc_auc(clf_rf, plt, traps, labels,'R Forest')

        print "Importances",clf.feature_importances_
        # print "OOB Score", clf.oob_score_
        # print "OOB Decision function", clf.oob_decision_function_
        print "Num features", clf.n_features_

        fontP = FontProperties()
        fontP.set_size('small')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        title = 'ROC with ' + str(file_num) + ' days of extra data'
        plt.title(title)
        plt.legend(loc="lower right", prop = fontP)
        # plt.show()
        file_name = '../plots/Week_Weather_Stratified#' + str(file_num) + '.jpg'
        plt.savefig(file_name)
        return clf

    # Obtains the predicted probabilities and saves to file
    def predict_virus(self, clf, days, in_file):
        in_file = actual_file_name(in_file, days)
        traps = pd.read_csv(in_file)
        id = traps['Id']
        traps = traps.drop('Id', 1)
        pred_prob = clf.predict_proba(traps)
        # print pred
        submission = pd.DataFrame()

        chance = []
        for list in pred_prob:
            chance.append(float("{0:.1f}".format(list[1])))
            # chance.append(float(list[1]))
        submission['Id'] = id
        submission['WnvPresent'] = chance
        file_name = '../submission/submission_' + str(days) + '.csv'
        submission.to_csv(file_name, index=False)

    # Saves the classifier to file
    def save_classifier(self, clf, name, num):
        import pickle
        pickle_name = "../models/" + name + "_" + str(num) + ".pickle"
        f = open(pickle_name, 'wb')
        pickle.dump(clf, f)
        f.close()


if __name__ == '__main__':
    day = 20
    in_train_file = '../input/train.csv'
    out_train_file = '../output/train_clean.csv'
    model = Model()
    model.clean_train(in_train_file, day, out=out_train_file)
    print "Cleaned train..."

    in_weather_file = '../input/weather.csv'
    out_weather_file = '../output/weather_corrected.csv'
    model.clean_weather(in_weather_file, day, out=out_weather_file)
    print "Cleaned weather..."

    out_merged_file = '../output/train_weather_spray.csv'
    model.merge_files(out_train_file, out_weather_file, day, out=out_merged_file)
    print "Merged train, weather, spray"

    out_train_appended = '../output/train_weather_spray_appended.csv'
    model.last_week_weather(day, out_merged_file, out_weather_file, out_train_appended)
    print "Added ", str(day), " days of past weather"

    out_clustered_file = '../output/train_weather_spray_clustered.csv'
    kmeans_locations_clf = model.cluster_locations(out_train_appended, day, out=out_clustered_file)
    print "Clustered location..."

    out_trap_clustered_file =  '../output/train_weather_spray_clustered_traps.csv'
    model.cluster_traps(out_clustered_file, day, out=out_trap_clustered_file)
    print "Clustered traps..."

    print "Classifying..."
    classifier = model.fit_model(day, out_trap_clustered_file)

    in_test_file = '../input/test.csv'
    out_test_file = '../output/test_clean.csv'
    model.clean_train(in_test_file, day, out=out_test_file)
    print "Cleaned test..."

    out_merged_test_file = '../output/test_weather_spray.csv'
    model.merge_files(out_test_file, out_weather_file, day, out=out_merged_test_file)
    print "Merged test, weather, spray"

    out_clustered_test_file = '../output/test_weather_spray_clustered.csv'
    model.cluster_locations(out_merged_test_file, day,  test=True, clf=kmeans_locations_clf, out=out_clustered_test_file)
    print "Clustered test data..."

    out_clustered_traps_test_file = '../output/test_weather_spray_clustered_traps.csv'
    model.cluster_traps(out_clustered_test_file, day, out=out_clustered_traps_test_file)
    print "Clustered test data..."

    print "Predicting..."
    model.predict_virus(classifier, day, out_clustered_traps_test_file)

    model.save_classifier(classifier, "Random_forest", day)
    model.save_classifier(kmeans_locations_clf, "k_means_locations", day)
