from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

## based in ##https://github.com/scikit-learn/scikit-learn/issues/7050
class VIPSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=20, vip_threshold=0.9):
        self.n_components = n_components
        self.vip_threshold = vip_threshold
        self.selected_mask_ = None

    def fit(self, X, y):
        if self.n_components is None:
            return self
        self.scaler_ = RobustScaler()
        X_ = self.scaler_.fit_transform(X)
        pls = PLSRegression(n_components=self.n_components)
        pls.fit(X_, y)
        t = pls.x_scores_
        w = pls.x_weights_
        q = pls.y_loadings_
        features_, _ = w.shape
        inner_sum = np.diag(t.T @ t @ q.T @ q)
        SS_total = np.sum(inner_sum)
        vip = np.sqrt(features_*(w**2 @ inner_sum)/SS_total)
        self.selected_mask_ = (vip > self.vip_threshold).ravel()
        return self

    def transform(self, X):
        if self.n_components is None:
            return X
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_mask_]
        else:
            return X[:, self.selected_mask_]
