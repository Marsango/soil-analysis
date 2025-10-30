import os
import traceback

import numpy as np
import pandas as pd
from pandas import DataFrame
from pybaselines import Baseline
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, IsolationForest, GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor


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

def apply_savgol_to_df(df, params):
    data = df.to_numpy(dtype=float)
    filtered = np.apply_along_axis(lambda x: savgol_filter(x, params["window_length"],
                                                           params["poly_order"], deriv=params["deriv"]), 1, data)
    return pd.DataFrame(filtered, columns=df.columns, index=df.index)

def apply_baseline_to_df(df):
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

def apply_normalization(df):
    fator = df.mean(axis=1)
    return df.div(fator, axis=0)

def handle_dataset_data(df):
    df = df.drop(df.columns[0], axis=1)
    # a lot of noise in the final of the signal, dropping wavelength > 1650
    # columns_to_drop = []
    # for column in df.columns:
    #     if column != "result" and 950 < float(column.replace(",", ".")) > 1600:
    #         columns_to_drop.append(column)
    # df = df.drop(columns=columns_to_drop)
    return df

##https://github.com/scikit-learn/scikit-learn/issues/7050
def calculate_vips(model):
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    features_, _ = w.shape
    inner_sum = np.diag(t.T @ t @ q.T @ q)
    SS_total = np.sum(inner_sum)
    vip = np.sqrt(features_*(w**2 @ inner_sum)/ SS_total)
    return vip

def pre_processing(X, params):
    # X = apply_detrend(X)
    # X = apply_baseline_to_df(X)
    # X = apply_msc(X)# seems good
    # X = apply_savgol_to_df(X, {'window_length': 5,
    #                                     'poly_order': 4, 'deriv': 0})

    # X = apply_baseline_to_df(X)
    X = apply_snv(X)
    X = apply_savgol_to_df(X, params)# change scales
    # X = apply_normalization(X) # till the moment doesnt work well
    return X


def remove_outliers(X, y, contamination=0.05):
    iso = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=300,
        bootstrap=True
    )
    preds = iso.fit_predict(X)
    mask = preds == 1

    X_clean = X.loc[mask]
    y_clean = y.loc[mask]
    print(f"Removed {np.sum(~mask)} outliers out of {len(mask)} samples")
    return X_clean, y_clean


def plot_row_graph(df, name):
    plt.figure(figsize=(10, 5))
    if isinstance(df, DataFrame):
        for i, row in df.iterrows():
            plt.plot(df.columns, row.values, label=f'Row {i}')
    else:
            plt.plot(df)

    plt.xlabel('Column index')
    plt.ylabel('Value')
    plt.title('Each Row of NumPy Array as a Line')
    plt.savefig(f"{name}.png", dpi=300)

def fit(X_train, y_train, X_test, y_test, pipeline):
    model = pipeline[0]
    model_name = pipeline[1]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} | RMSE: {rmse} | R²: {r2}")
    return rmse, r2

def load_datasets():
    files = os.listdir("generated_datasets")
    return [f for f in files if "187_samples" in f]

def create_pipelines(X_train, y_train):
    pipeline_list = []
    # svr_best_model = svr_tunning(X_train, y_train)
    #
    # # pipeline_list.append((svr_best_model, "SVR"))
    # # pipeline_list.append((plsr_tunning(X_train, y_train), "PLSR"))
    # # pipeline_list.append((random_forest_tuning(X_train, y_train), "RandomForest"))
    # # pipeline_list.append((gbr_tuning(X_train, y_train), "GradientBoost"))
    pipeline_list.append((xgb_tuning(X_train, y_train), "XGBoost"))
    return pipeline_list

def plsr_tunning(X_train, y_train):
    param_dist = {
        'plsregression__n_components': [n for n in range(2, 21)],
    }

    search = GridSearchCV(make_pipeline(RobustScaler(), PLSRegression()), param_dist, cv=5, n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_


def svr_tunning(X_train, y_train):
    pipe_svr = Pipeline([
        ('scaler', RobustScaler()),
        ('model', SVR())
    ])

    param_grid_rbf = {
        'model__kernel': ['rbf'],
        'model__C': np.logspace(-1, 3, 5),
        'model__gamma': np.logspace(-4, 0, 5)
    }

    param_grid_linear = {
        'model__kernel': ['linear'],
        'model__C': np.logspace(-3, 2, 6),
    }

    search_rbf = GridSearchCV(pipe_svr, param_grid_rbf, cv=5, n_jobs=-1, scoring='r2')
    search_rbf.fit(X_train, y_train)

    search_linear = GridSearchCV(pipe_svr, param_grid_linear, cv=5, n_jobs=-1, scoring='r2')
    search_linear.fit(X_train, y_train)

    print(f"Best SVR (RBF) params: {search_rbf.best_params_} (R2: {search_rbf.best_score_:.4f})")
    print(f"Best SVR (Linear) params: {search_linear.best_params_} (R2: {search_linear.best_score_:.4f})")
    if search_rbf.best_score_ > search_linear.best_score_:
        print("Best SVR: rbf")
        return search_rbf.best_estimator_
    else:
        print("Best SVR: linear")
        return search_linear.best_estimator_

    search = GridSearchCV(pipe_svr, param_grid, cv=5, n_jobs=-1, scoring='r2')
    search.fit(X_train, y_train)

    print(f"Best SVR params: {search.best_params_}")
    return search.best_estimator_

def gbr_tuning(X_train, y_train):
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 8, 10],
        'subsample': [0.7, 0.9, 1.0],
        'max_features': ['sqrt', 'log2', None]
    }
    search = RandomizedSearchCV(
        estimator=GradientBoostingRegressor(random_state=42),
        param_distributions=param_dist,
        n_iter=15,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        random_state=42
    )
    search.fit(X_train, y_train)
    return search.best_estimator_

def random_forest_tuning(X_train, y_train):
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_features': ['sqrt', 'log2', 0.8],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_distributions=param_dist,
        n_iter=25,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        random_state=42
    )
    search.fit(X_train, y_train)
    return search.best_estimator_

from xgboost import XGBRegressor

def xgb_tuning(X_train, y_train):
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.7, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2]
    }

    search = RandomizedSearchCV(
        estimator=XGBRegressor(random_state=42),
        param_distributions=param_dist,
        n_iter=25,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train)

    return search.best_estimator_

if __name__ == '__main__':
        for dataset_name in load_datasets():
            if not dataset_name.startswith("MO_"):
                continue
            window_size = [5, 11, 21, 31]
            poly_order = [1, 2, 3, 4]
            deriv = [0, 1, 2]
            best_r2 = {'params': {}, 'score': -10, "model": None}
            best_rmse = {'params': {}, 'score': 1000, "model": None}
            # for w in window_size:
            #     for p in poly_order:
            #         for d in deriv:
            try:
                params = {"window_length": 31, "poly_order": 2, "deriv": 0}
                print("\n--------------------" + dataset_name + "--------------------\n")
                print(f"\n SG PARAMS: {params}")
                dataframe = pd.read_csv("./generated_datasets/" + dataset_name, decimal='.')
                df = handle_dataset_data(dataframe)
                y = df['result'].copy()
                X = df.drop('result', axis=1)
                plot_row_graph(y, "result_raw")
                plot_row_graph(X, "nir_raw")

                q_low = df['result'].quantile(0.025)
                q_high = df['result'].quantile(0.975)


                df = df[(df['result'] >= q_low) & (df['result'] <= q_high)]
                y = df['result'].copy()
                X = df.drop('result', axis=1)
                print(f"Shape after removing y-outliers: {df.shape}")
                X = pre_processing(X, params)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                # scaler = RobustScaler()
                # X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns,
                #                               index=X_train.index)
                # X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
                # plsr = plsr_tunning(X_train, y_train).named_steps['plsregression']
                # vip_scores = calculate_vips(plsr)
                # important_features_mask = vip_scores > 1
                # X_train = X_train.loc[:, important_features_mask]
                # X_test = X_test.loc[:, important_features_mask]

                # X_train, y_train = remove_outliers(X_train, y_train)
                print(X_train.shape)
                plot_row_graph(X_train, "train_data")
                plot_row_graph(X_test, "test_data")

                pipelines = create_pipelines(X_train, y_train)
                for pipeline in pipelines:
                    rmse, r2 = fit(X_train, y_train, X_test, y_test, pipeline)
                    if rmse < best_rmse["score"]:
                        best_rmse["score"]  = rmse
                        best_rmse["param"] = params
                        best_rmse["model"] = pipeline[1]
                    if r2 > best_r2["score"]:
                        best_r2["score"]  = r2
                        best_r2["param"] = params
                        best_r2["model"] = pipeline[1]
            except:
                print(f"Error with parameters {params}")
                traceback.print_exc()
                continue
            print(f"Best R2 score:  {best_r2}")
            print(f"Best RMSE score:  {best_rmse}")