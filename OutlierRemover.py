
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
import pandas as pd

class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, contamination=0.05, n_estimators=300, bootstrap=True, random_state=42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.iso = None
        self.mask_ = None

    def fit(self, X, y=None):
        self.iso = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            bootstrap=self.bootstrap
        )
        self.iso.fit(X)
        preds = self.iso.predict(X)
        self.mask_ = preds == 1
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X_clean = X.loc[self.mask_]
        else:
            X_clean = X[self.mask_]

        if y is not None:
            if isinstance(y, (pd.Series, pd.DataFrame)):
                y_clean = y.loc[self.mask_]
            else:
                y_clean = y[self.mask_]
            return X_clean, y_clean

        return X_clean
