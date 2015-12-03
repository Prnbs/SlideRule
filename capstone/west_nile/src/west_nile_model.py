__author__ = 'psinha4'

import pandas as pd
import numpy as np
import datetime
from sklearn.cluster import KMeans
from matplotlib.font_manager import FontProperties

from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import KFold, StratifiedKFold


def clean_train(in_file, out_file):
    traps = pd.read_csv(in_file, parse_dates=['Date'])
    d_categorical_species = {}
    d_categorical_species['CULEX ERRATICUS'] = 0
    d_categorical_species['CULEX PIPIENS'] = 1
    d_categorical_species['CULEX PIPIENS/RESTUANS'] = 2
    d_categorical_species['CULEX RESTUANS'] = 3
    d_categorical_species['CULEX SALINARIUS'] = 4
    d_categorical_species['CULEX TARSALIS'] = 5
    d_categorical_species['CULEX TERRITANS'] = 6
    d_categorical_species['UNSPECIFIED CULEX'] = 7

    species_categorical = np.full((len(traps), 1), 0)

    species_idx = traps.columns.get_loc("Species")

    for index, row in traps.iterrows():
        if d_categorical_species[traps.iloc[index, species_idx]] == 1 or \
            d_categorical_species[traps.iloc[index, species_idx]] == 3:
            species_categorical[index] = 1
        elif d_categorical_species[traps.iloc[index, species_idx]] == 2:
            species_categorical[index] = 1
        else:
            species_categorical[index] = 0

    traps['IsPipiens'        ] = ((traps['Species']=='CULEX PIPIENS'  )*1 +           # 8.9%   / 2699
                               (traps['Species']=='CULEX PIPIENS/RESTUANS')*0.5)
    traps['IsPipiensRestuans'] = ((traps['Species']=='CULEX PIPIENS/RESTUANS')*1 +    # 5.5%   / 4752
                               (traps['Species']=='CULEX PIPIENS'  )*0 + (traps['Species']=='CULEX RESTUANS'  )*0)
    traps['IsRestuans'       ] = ((traps['Species']=='CULEX RESTUANS'  )*1 +          # 1.8%   / 2740
                               (traps['Species']=='CULEX PIPIENS/RESTUANS')*0.5)
    traps['IsOther'          ] = (traps['Species']!='CULEX PIPIENS')*(traps['Species']!='CULEX PIPIENS/RESTUANS')*(traps['Species']!='CULEX RESTUANS')*1


    traps = traps.drop("Species", 1)
    traps = traps.drop("Address", 1)
    traps = traps.drop("Block", 1)
    traps = traps.drop("Street", 1)
    traps = traps.drop("AddressNumberAndStreet", 1)
    traps = traps.drop("AddressAccuracy", 1)
    traps = traps.drop("Trap", 1)

    # traps['Species_Categorical'] = species_categorical

    traps.to_csv(out_file, index=False)


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


def clean_weather(in_file, out_file):
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

    weather.to_csv(out_file, index=False)


def merge_files(in_train_file, in_weather_file, out_file):
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
                    # print  (date - dates_sprayed[index]).days
                    last_sprayed[trap_idx_last_item] =  (date - dates_sprayed[index]).days
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
    merged.to_csv(out_file, index=False)


def last_week_weather(days_past, in_merged_file, in_weather_file, out_file):
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
        data_from_traps = row[1:10]
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


def cluster_locations(in_file, out_file, test=False, clf=None):
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


def plot_roc_auc(clf, plt, traps, labels, name):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Compute micro-average ROC curve and ROC area

    kf = StratifiedKFold(labels, n_folds=10, shuffle=True)

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
             label=label % mean_auc, lw=2)
    return plt


def roc_auc(file_num, in_file):
    traps = pd.read_csv(in_file)

    # extract label
    labels = traps['WnvPresent']
    # drop label from features
    traps.drop('WnvPresent', 1, inplace=True)

    from sklearn import tree
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC

    clf_rf = RandomForestClassifier(n_estimators=100)
    clf_nb = GaussianNB()
    clf_svc = SVC(kernel='linear', C=1000.0)
    clf_dt = tree.DecisionTreeClassifier(criterion='gini',min_samples_split=100)
    # Compute ROC curve and ROC area for each class
    import matplotlib.pyplot as plt
    plt.clf()
    # plt = plot_roc_auc(clf_dt, plt, traps, labels,'D Tree')
    # plt = plot_roc_auc(clf_nb, plt, traps, labels,'G NB')
    plt = plot_roc_auc(clf_rf, plt, traps, labels,'R Forest')
    # plt = plot_roc_auc(clf_svc, plt, traps, labels,'SVC')

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
    return clf_rf


def predict_virus(clf, in_file, days):
    traps = pd.read_csv(in_file)
    id = traps['Id']
    traps = traps.drop('Id', 1)
    pred = clf.predict(traps)
    pred_prob = clf.predict_proba(traps)
    # print pred
    submission = pd.DataFrame()
    traps['WnvPresent'] = pred

    chance = []
    for list in pred_prob:
        chance.append(float("{0:.1f}".format(list[1])))
    submission['Id'] = id
    submission['WnvPresent'] = chance
    file_name = '../output/submission_' + str(day) + '.csv'
    submission.to_csv(file_name, index=False)


def save_classifier(clf, name, num):
    import pickle
    pickle_name = "../models/" + name + "_" + str(num) + ".pickle"
    f = open(pickle_name, 'wb')
    pickle.dump(clf, f)
    f.close()


if __name__ == '__main__':
    in_train_file = '../input/train.csv'
    out_train_file = '../output/train_clean.csv'
    clean_train(in_train_file, out_train_file)
    print "Cleaned train..."

    in_weather_file = '../input/weather.csv'
    out_weather_file = '../output/weather_corrected.csv'
    clean_weather(in_weather_file, out_weather_file)
    print "Cleaned weather..."

    out_merged_file = '../output/train_weather_spray.csv'
    merge_files(out_train_file, out_weather_file, out_merged_file)
    print "Merged train, weather, spray"

    # days_past = [5,7,10,15,20]
    # for day in days_past:

    day = 20
    out_train_appended = '../output/train_weather_spray_appended.csv'
    last_week_weather(day, out_merged_file, out_weather_file, out_train_appended)
    print "Added ", str(day), " days of past weather"

    out_clustered_file = '../output/train_weather_spray_clustered.csv'
    kmeans_clf = cluster_locations(out_train_appended, out_clustered_file)
    print "Clustered data..."

    classifier = roc_auc(day, out_clustered_file)

    print classifier.feature_importances_

    in_test_file = '../input/test.csv'
    out_test_file = '../output/test_clean.csv'
    clean_train(in_test_file, out_test_file)
    print "Cleaned test..."

    out_merged_test_file = '../output/test_weather_spray.csv'
    merge_files(out_test_file,out_weather_file, out_merged_test_file)
    print "Merged test, weather, spray"

    out_clustered_test_file = '../output/test_weather_spray_clustered.csv'
    cluster_locations(out_merged_test_file, out_clustered_test_file, test=True, clf=kmeans_clf)
    print "Clustered test data..."

    print "Predicting..."
    predict_virus(classifier,out_clustered_test_file, day)

    save_classifier(classifier,"Random_forest", day)
    save_classifier(kmeans_clf,"k_means", day)
