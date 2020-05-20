

__author__ = "Saswati Ray"
__email__ = "sray@cs.cmu.edu"

import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from tpot import TPOTClassifier
import pandas as pd 
import numpy as np
import openml as oml
import sys
import liteautoml, evaluate_hyperopt

import warnings
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

n_jobs = 8
timeout=int(sys.argv[1])
scorefile=sys.argv[2]

outfile = open(scorefile,"a")

ids = [sys.argv[3]]
for id in ids:
    print("running for id = ", id)
    dataset = oml.datasets.get_dataset(id)
    X, y, cat_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)
    ydf = pd.DataFrame(y)
    rows = len(X)
    classes = len(ydf.iloc[:,0].value_counts())
    frequencies = ydf.iloc[:,0].value_counts()
    dummy = frequencies.iat[0]/len(X)

    feat_type = ['Categorical' if ci else 'Numerical'
                 for ci in cat_indicator]

    ### Label Encoding
    encoder = preprocessing.LabelEncoder()
    try:
        encoder.fit(y)
        y = encoder.transform(y)
    except:
        print(sys.exc_info()[0])
        continue

    imputer = SimpleImputer(strategy='most_frequent')
    cat_count = 0
    # Label encode categorical attributes and impute missing values 
    for i in range(len(attribute_names)):
        if cat_indicator[i] == True:
            try:
                values = np.reshape(X.iloc[:,i].values, (-1,1))
                values = imputer.fit_transform(values)
                encoder.fit(values)
                train = encoder.transform(values)
                X[attribute_names[i]] = train
                cat_count = cat_count + 1
            except:
                print(sys.exc_info()[0])
                continue

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)

    timeout_in_sec = timeout*60

    # Run auto-sklearn framework on the dataset
    auto_score = -1
    try:
        automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=timeout_in_sec, n_jobs=n_jobs)
        automl.fit(X_train, y_train, feat_type=feat_type)
        #automl.fit_ensemble(y_train, ensemble_size=50)
        y_hat = automl.predict(X_test.values)
        auto_score = accuracy_score(y_test, y_hat)
    except:
        print(sys.exc_info()[0])

    # Run TPOT framework on the dataset
    tpot_score = -1
    try:
        tpot = TPOTClassifier(verbosity=0, n_jobs=n_jobs, random_state=1, max_time_mins=timeout, max_eval_time_mins=0.04, population_size=15)
        tpot.fit(X_train, y_train)
        tpot_score = tpot.score(X_test, y_test)
    except:
        print(sys.exc_info()[0])

    # Run Lite-AutoML framework on the dataset
    (best, atts, cl) = liteautoml.compute_score(X_train, y_train, X_test, y_test, cat_indicator, n_jobs, timeout_in_sec)
    hp_best = evaluate_hyperopt.compute_score(X_train, y_train, X_test, y_test, cat_indicator, n_jobs, timeout_in_sec)
    outfile.write(dataset.name + "," + str(dummy) + "," + cl + "," + str(id) + "," + str(rows) + "," + str(classes) + "," + str(auto_score) + "," + str(tpot_score) + "," + str(hp_best) + "," + str(best) +  "," + str(len(X.columns)) + "," + str(atts) +'\n')
outfile.close()
