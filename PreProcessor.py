from pybaselines import Baseline
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

def apply_detrend(df):

    data = df.to_numpy(dtype=float)
    n_samples, n_features = data.shape

    x = np.arange(n_features)

    output_data = np.zeros_like(data)

    for i in range(n_samples):
        row = data[i, :]
        poly_coeffs = np.polyfit(x, row, 2)
        baseline = np.polyval(poly_coeffs, x)
        output_data[i, :] = row - baseline

    return pd.DataFrame(output_data, columns=df.columns, index=df.index)

def apply_msc(df, reference_spectrum=None, eps=1e-8):
    data = df.to_numpy(dtype=float)
    n_samples, n_wavelengths = data.shape

    if reference_spectrum is None:
        ref = data.mean(axis=0)
    elif reference_spectrum == 'median':
        ref = df.median(axis=0)

    output_data = np.zeros_like(data, dtype=float)
    for i in range(n_samples):
        row = data[i, :].astype(float)
        A = np.vstack([np.ones(n_wavelengths), ref]).T
        coef, *_ = np.linalg.lstsq(A, row, rcond=None)
        a, b = coef
        if abs(b) < eps:
            b = 1.0
        output_data[i, :] = (row - a) / b

    corrected_df = pd.DataFrame(output_data, columns=df.columns, index=df.index)
    return corrected_df

def apply_asls_baseline_to_df(df):
    baseline_fitter = Baseline()
    data = df.to_numpy(dtype=float)

    corrected = np.zeros_like(data)
    for i, row in enumerate(data):
        baseline, _ = baseline_fitter.asls(row, lam=1e6, p=0.005)
        corrected[i] = row - baseline

    return pd.DataFrame(corrected, columns=df.columns, index=df.index)

def apply_snv(df):
    output_data = np.zeros_like(df, dtype=float)
    for i in range(df.shape[0]):
        row = df.iloc[i, :].to_numpy(dtype=float)
        output_data[i, :] = (row - np.mean(row)) / np.std(row)
    return pd.DataFrame(output_data, columns=df.columns, index=df.index)

def apply_savgol_to_df(df, params):
    data = df.to_numpy(dtype=float)
    filtered = np.apply_along_axis(lambda x: savgol_filter(x, params["window_length"],
                                                           params["poly_order"], deriv=params["deriv"]), 1, data)
    return pd.DataFrame(filtered, columns=df.columns, index=df.index)

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, scatter_correction='snv', baseline_correction=None,
                 sg_window=5, sg_poly=2, sg_deriv=0, sg_enabled=True):
        self.scatter_correction = scatter_correction
        self.sg_enabled = sg_enabled
        self.baseline_correction = baseline_correction
        self.sg_window = sg_window
        self.sg_poly = sg_poly
        self.sg_deriv = sg_deriv

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_processed = X.copy()

        if self.scatter_correction == 'snv':
            X_processed = apply_snv(X_processed)
        elif self.scatter_correction == 'msc':
            X_processed = apply_msc(X_processed)

        if self.baseline_correction == 'asls':
            X_processed = apply_asls_baseline_to_df(X_processed)
        elif self.baseline_correction == 'detrend':
            X_processed = apply_detrend(X_processed)

        if self.sg_enabled:
            return X_processed

        params = {"window_length": self.sg_window,
                  "poly_order": self.sg_poly,
                  "deriv": self.sg_deriv}

        if self.sg_poly < self.sg_deriv:
            return X_processed

        X_processed = apply_savgol_to_df(X_processed, params)
        return X_processed