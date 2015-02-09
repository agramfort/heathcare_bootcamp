import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import LabelBinarizer
from scipy import sparse


class TreeTransform(BaseEstimator, TransformerMixin):

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        self.binarizers_ = []
        sparse_applications = []
        estimators = np.asarray(self.estimator_.estimators_).ravel()
        for t in estimators:
            lb = LabelBinarizer(sparse_output=True)
            sparse_applications.append(lb.fit_transform(t.tree_.apply(X)))
            self.binarizers_.append(lb)
        return sparse.hstack(sparse_applications)

    def transform(self, X, y=None):
        sparse_applications = []
        estimators = np.asarray(self.estimator_.estimators_).ravel()
        for t, lb in zip(estimators, self.binarizers_):
            sparse_applications.append(lb.transform(t.tree_.apply(X)))
        return sparse.hstack(sparse_applications)


def model(X_train, y_train, X_test):
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    trees = RandomForestClassifier(
        max_depth=10, n_estimators=300, random_state=0,
    )

    clf = make_pipeline(
        Imputer(strategy='most_frequent'),
        TreeTransform(trees),
        LogisticRegression(C=1.0, fit_intercept=True)
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)
    return y_pred, y_score
