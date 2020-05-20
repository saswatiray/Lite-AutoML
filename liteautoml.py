

__author__ = "Saswati Ray"
__email__ = "sray@cs.cmu.edu"

from sklearn.naive_bayes import BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, KFold
from multiprocessing import Pool
from sklearn.metrics import accuracy_score
from timeit import default_timer as timer
import numpy as np
import pandas as pd
import sys, math, random

def get_num_splits(length):
    """
    Return the number of splits for k-fold CV.
    """
    splits = 2
    if length < 500:
        splits = 50
        if length < splits:
            splits = length
    elif length < 1000:
        splits = 25
    elif length < 2500:
        splits = 20
    elif length < 5000:
        splits = 10
    elif length < 10000:
        splits = 5
    elif length < 20000:
        splits = 3
    else:
        splits = 2
    return splits

def score_solution(Xcv, ycv, model):
    """
    Run k-fold CV
    """
    splits = get_num_splits(len(Xcv))
    if Xcv.shape[1] >= 500:
        splits = max(2, int(splits*0.5))
    frequencies = ycv.iloc[:,0].value_counts()
    min_freq = frequencies.iat[len(frequencies)-1]
    if min_freq < splits:
        kf = KFold(n_splits=splits, shuffle=True, random_state=9001)
        split_indices = kf.split(Xcv)
    else:
        kf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=9001)
        split_indices = kf.split(Xcv, ycv)
    metric_sum = 0
    # Do the actual k-fold CV here
    for train_index, test_index in split_indices:
        X1, X2 = Xcv.iloc[train_index,:], Xcv.iloc[test_index,:]
        y1, y2 = ycv.iloc[train_index,:], ycv.iloc[test_index,:]
        model.fit(X1, y1)
        predictions = model.predict(X2)
        metric = accuracy_score(y2, predictions)
        metric_sum += metric
    score = metric_sum/splits
    return score

def reduce_dimensionality(X_train, X_test, cat_indicator):
    """
    Attribute sub-sampling
    """
    num_atts = X_train.shape[1]
    if num_atts <= 1000:
        return (X_train, X_test, cat_indicator)

    cols = np.arange(num_atts)
    random.seed(1)
    cols = random.sample(list(cols), 1000)

    X_train = X_train.iloc[:,cols]
    X_test = X_test.iloc[:,cols]
    cat_indicator = np.array(cat_indicator)[cols]
    return (X_train, X_test, cat_indicator)

def compute_score(X_train, y_train, X_test, y_test, cat_indicator, n_jobs, orig_timeout):
    trees = 100
    max_iter = 1000
    if len(X_train) >= 100000:
        trees = 10
        max_iter = 100

    print("Start!")
    timeout = orig_timeout
    start = timer()
 
    (X_train, X_test, cat_indicator) = reduce_dimensionality(X_train, X_test, cat_indicator)

    classifiers = [BernoulliNB(), LinearDiscriminantAnalysis(), LogisticRegression(random_state=1), AdaBoostClassifier(random_state=1), LinearSVC(max_iter=max_iter, random_state=1), ExtraTreesClassifier(random_state=1, n_estimators=trees), RandomForestClassifier(random_state=1, n_estimators=trees), BaggingClassifier(random_state=1, n_estimators=10), MLPClassifier(random_state=1,early_stopping=True), GradientBoostingClassifier(max_features=5, random_state=1, n_estimators=10)]
    
    model_steps = [SimpleImputer(strategy='median'), RobustScaler()]
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

    cats = []
    rows = len(X_train)
    # use rule of thumb to exclude categorical atts with high cardinality for one-hot-encoding
    max_num_cols = math.log(rows, 2)
    if rows > 100000:
        max_num_cols = max_num_cols/4
    
    # Iterate over all categorical attributes
    for i in range(len(cat_indicator)):
        if cat_indicator[i] is True:
            arity = len(X_train.iloc[:,i].unique())
            if arity <= max_num_cols:
                cats.append(i)

    if len(cats) > 0:
        start1=timer()
        X_train.reset_index(drop=True,inplace=True)
        X_object = X_train.iloc[:,cats]
        codes = ohe.fit_transform(X_object)
        X_train = pd.concat([X_train.drop(X_train.columns[cats],axis=1),
               pd.DataFrame(codes).astype(int)], axis=1)
        end1=timer()
    
    for m in model_steps:
        X_train = m.fit_transform(X_train)

    y_train = pd.DataFrame(y_train)
    X_train = pd.DataFrame(X_train)

    num_atts = X_train.shape[1]
    if num_atts <= 50 and rows <= 10000:
        classifiers.append(KNeighborsClassifier(n_neighbors=10))

    # For ensembles
    if num_atts >= 500:
        classifiers[5].max_features="log2"
        classifiers[6].max_features="log2"
        classifiers[7].max_features=0.8
    classifiers[9].max_features=min(5, num_atts) 

    # For bagging
    if num_atts < 100:
        if rows <= 10000:
            classifiers[7].n_estimators = 100
        elif rows <= 50000:
            classifiers[7].n_estimators = 50
        else:
            classifiers[7].n_estimators = 10

    async_message_thread = Pool((int)(n_jobs))
    results = [async_message_thread.apply_async(score_solution, (X_train, y_train, c)) for c in classifiers]
    index = 0
    scores = []

    end = timer()
    time_used = end - start
    timeout = timeout - time_used
    print("time remaining = ", timeout)
    for r in results:
        try:
            start_solution = timer()
            score = r.get(timeout=timeout)
            scores.append(score)
            end_solution = timer()
            time_used = end_solution - start_solution
            timeout = timeout - time_used
            if timeout <= 0:
                 timeout = 3
        except TimeoutError:
            timeout = 1
        except:
            print(sys.exc_info()[0])
            print("Solution terminated: ", classifiers[index])
            print(X_train.shape)
            scores.append(-1)
            end_solution = timer()
            time_used = end_solution - start_solution
            timeout = timeout - time_used
            if timeout <= 0:
                 timeout = 1
        index = index + 1

    pca = None
    RFpca = None
    print("time remaining = ", timeout)
    if timeout >= 10 and len(X_train) < 100000:
        from sklearn.decomposition import PCA
        start_solution = timer()
        n_comp = min(10, X_train.shape[1])
        pca = PCA(n_components=n_comp)
        Xpca = pca.fit_transform(X_train)
        end_solution = timer()
        time_used = end_solution - start_solution
        print("PCA = ", time_used)
        RFpca = RandomForestClassifier(random_state=1)
        score = score_solution(pd.DataFrame(Xpca), y_train, RFpca)
        scores.append(score)
        classifiers.append(RFpca)
    else:
        classifiers.append(None)
        scores.append(-1)

    timeout = timeout - time_used
    bagged_trees = classifiers[7].n_estimators
    while timeout > 0.1 * orig_timeout:
        trees = trees + 100
        print("Trying trees = ", trees)
        classifiers.append(ExtraTreesClassifier(random_state=1, n_estimators=trees))
        classifiers.append(RandomForestClassifier(random_state=1, n_estimators=trees))
        bagged_trees = bagged_trees + 10
        classifiers.append(BaggingClassifier(random_state=1, max_features=classifiers[7].max_features, n_estimators=bagged_trees))
        
        results = [async_message_thread.apply_async(score_solution, (X_train, y_train, c)) for c in classifiers[10:13]]
        for r in results:
            try:
                start_solution = timer()
                score = r.get(timeout=timeout)
                scores.append(score)
                end_solution = timer()
                time_used = end_solution - start_solution
                timeout = timeout - time_used
                if timeout <= 0:
                    timeout = 1
            except TimeoutError:
                timeout = 1
            except:
                print(sys.exc_info()[0])
                print("Solution terminated: ")
                scores.append(-1)
                end_solution = timer()
                time_used = end_solution - start_solution
                timeout = timeout - time_used
                if timeout <= 0:
                    timeout = 1
        if trees > 1000:
            break

    print(scores)
    # Sort solutions by their scores and rank them
    sorted_x = np.argsort(scores)
    best_model = None
    bestindex = sorted_x[len(scores)-1]

    if bestindex == 10:
        # Best is PCA-RF model
        best_model = RFpca
        best_model.fit(Xpca, y_train)
        model_steps.append(pca)
        cl = "pca+rf"
    else:
        best_model = classifiers[bestindex]
        print(best_model)
        best_model.fit(X_train, y_train)
        cl = type(best_model).__name__

    if len(cats) > 0:
       # OHE
        X_test.reset_index(drop=True,inplace=True)
        X_object = X_test.iloc[:,cats]
        codes = ohe.transform(X_object)
        X_test = pd.concat([X_test.drop(X_test.columns[cats],axis=1),
               pd.DataFrame(codes).astype(int)], axis=1)

    for m in model_steps:
        X_test = m.transform(X_test)

    y_hat = best_model.predict(X_test)
    best = accuracy_score(y_test, y_hat)
    #for c in classifiers:
    #    c.fit(X_train, y_train)
    #    y_hat = c.predict(X_test)
    #    best1 = accuracy_score(y_test, y_hat)
    #    print(c)
    #    print(best1)
    return (best, len(X_train.columns), cl)

