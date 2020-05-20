

__author__ = "Saswati Ray"
__email__ = "sray@cs.cmu.edu"

from hpsklearn import HyperoptEstimator, any_classifier
from hyperopt import tpe
import numpy as np

def compute_score(X_train, y_train, X_test, y_test, cat_indicator, n_jobs, timeout):
    estim = HyperoptEstimator(classifier=any_classifier('clf'), algo=tpe.suggest, max_evals=60, trial_timeout=timeout/60)
    best = -1
    try:
        estim.fit( X_train, y_train)
        best = estim.score(X_test, y_test)
        print(estim.best_model())
    except:
        best = -1
    return best
