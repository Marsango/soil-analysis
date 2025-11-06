import os

import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, IsolationForest, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from PreProcessor import Preprocessor
from VIPSelector import VIPSelector



def handle_dataset_data(df):
    df = df.drop(df.columns[0], axis=1)
    # a lot of noise in the final of the signal, dropping wavelength > 1650
    # columns_to_drop = []
    # for column in df.columns:
    #     if column != "result" and 950 < float(column.replace(",", ".")) > 1600:
    #         columns_to_drop.append(column)
    # df = df.drop(columns=columns_to_drop)
    y = df['result'].copy()
    X = df.drop('result', axis=1)
    plot_Y(y, "y_raw")
    plot_X(X, "X_raw")
    return df

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
    return X_clean, y_clean


def plot_X(df, name):
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

def plot_Y(df, name):
    y_sorted = np.sort(df)
    x_indices = np.arange(len(y_sorted))
    plt.figure(figsize=(10, 6))

    plt.scatter(x_indices, y_sorted)
    plt.title('Sorted Y-Values Plot')
    plt.xlabel('Sample Index')
    plt.ylabel('Y-Value')
    plt.grid(True)
    plt.savefig(f"{name}.png", dpi=300)


def fit(X, y, pipeline, outer_cv):
    model = pipeline[0]
    model_name = pipeline[1]

    scoring_metrics = {
        'r2': 'r2',
        'rmse': 'neg_root_mean_squared_error'
    }

    scores = cross_validate(model, X, y, cv=outer_cv,
                            scoring=scoring_metrics, n_jobs=-1)
    scores_r2 = scores['test_r2']
    scores_rmse = scores['test_rmse']

    r2_mean = scores_r2.mean()
    r2_std = scores_r2.std()

    rmse_mean = -scores_rmse.mean()
    rmse_std = scores_rmse.std()

    print(f"{model_name} | RMSE: mean: {rmse_mean:.4f} std: {rmse_std:.4f} | R²: mean: {r2_mean:.4f} std: {r2_std:.4f}")

    return rmse_mean, r2_mean

def load_datasets():
    files = os.listdir("generated_datasets")
    return [f for f in files if "187_samples" in f]

def create_pipelines(inner_cv):
    pipeline_list = []
    svr_best_model = svr_tuning(inner_cv)

    pipeline_list.extend(svr_best_model)
    pipeline_list.append((plsr_tuning(inner_cv), "PLSR"))
    pipeline_list.append((random_forest_tuning(inner_cv), "RandomForest"))
    pipeline_list.append((gbr_tuning(inner_cv), "GradientBoost"))
    pipeline_list.append((xgb_tuning(inner_cv), "XGBoost"))
    return pipeline_list

def plsr_tuning(inner_cv):

    pipeline = Pipeline([
        # ('outlier_remover', OutlierRemover()),
        ('preprocessor', Preprocessor()),
        ('scaler', RobustScaler()),
        ('vip', VIPSelector()),
        ('model', PLSRegression())
    ])

    param_dist = {
        'preprocessor__scatter_correction': ['snv', 'msc', None],
        'preprocessor__baseline_correction': ['asls', 'detrend', None],
        'preprocessor__sg_window': [5, 11, 21, 31],
        'preprocessor__sg_poly': [1, 2, 3, 4],
        'preprocessor__sg_deriv': [0, 1, 2],

        'model__n_components': [5, 10, 15, 20],

        'vip__n_components': [5, 10, 15, 20, None],
        'vip__vip_threshold': [0.8, 0.9, 1, 1.1]
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=500,
        cv=inner_cv,
        scoring='r2',
        n_jobs=-1,
        random_state=42
    )

    return search


def svr_tuning(inner_cv):
    pipe_svr = Pipeline([
        # ('outlier_remover', OutlierRemover()),
        ('preprocessor', Preprocessor()),
        ('scaler', RobustScaler()),
        ('vip', VIPSelector()),
        ('model', SVR())
    ])

    param_grid_rbf = {
        'preprocessor__scatter_correction': ['snv', 'msc', None],
        'preprocessor__baseline_correction': ['asls', 'detrend', None],
        'preprocessor__sg_window': [5, 11, 21, 31],
        'preprocessor__sg_poly': [1, 2, 3, 4],
        'preprocessor__sg_deriv': [0, 1, 2],

        'model__kernel': ['rbf'],
        'model__C': np.logspace(-1, 3, 5),
        'model__gamma': np.logspace(-4, 0, 5),

        'vip__n_components': [5, 10, 15, 20, None],
        'vip__vip_threshold': [0.8, 0.9, 1, 1.1]
    }

    param_grid_linear = {
        'preprocessor__scatter_correction': ['snv', 'msc', None],
        'preprocessor__baseline_correction': ['asls', 'detrend', None],
        'preprocessor__sg_window': [5, 11, 21, 31],
        'preprocessor__sg_poly': [1, 2, 3, 4],
        'preprocessor__sg_deriv': [0, 1, 2],

        'model__kernel': ['linear'],
        'model__C': np.logspace(-3, 2, 6),
        'vip__n_components': [5, 10, 15, 20, None],
        'vip__vip_threshold': [0.8, 0.9, 1, 1.1]
    }

    search_rbf = RandomizedSearchCV(pipe_svr, param_grid_rbf, cv=inner_cv, n_jobs=-1, scoring='r2', random_state=42, n_iter=500)
    search_linear = RandomizedSearchCV(pipe_svr, param_grid_linear, cv=inner_cv, n_jobs=-1, scoring='r2', random_state=42, n_iter=500)
    return (search_rbf, "SVR_RBF"), (search_linear, "SVR_linear")


def gbr_tuning(inner_cv):
    pipeline = Pipeline([
        # ('outlier_remover', OutlierRemover()),
        ('preprocessor', Preprocessor()),
        ('scaler', RobustScaler()),
        ('vip', VIPSelector()),
        ('model', GradientBoostingRegressor(random_state=42))
    ])

    param_dist = {
        'preprocessor__scatter_correction': ['snv', 'msc', None],
        'preprocessor__baseline_correction': ['asls', 'detrend', None],
        'preprocessor__sg_window': [5, 11, 21, 31],
        'preprocessor__sg_poly': [1, 2, 3, 4],
        'preprocessor__sg_deriv': [0, 1, 2],

        'model__n_estimators': [100, 200, 300, 500],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__max_depth': [3, 5, 8, 10],
        'model__subsample': [0.7, 0.9, 1.0],
        'model__max_features': ['sqrt', 'log2', None],

        'vip__n_components': [5, 10, 15, 20, None],
        'vip__vip_threshold': [0.8, 0.9, 1, 1.1]
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=500,
        cv=inner_cv,
        scoring='r2',
        n_jobs=-1,
        random_state=42
    )

    return search

def random_forest_tuning(inner_cv):
    pipeline = Pipeline([
        # ('outlier_remover', OutlierRemover()),
        ('preprocessor', Preprocessor()),
        ('scaler', RobustScaler()),
        ('vip', VIPSelector()),
        ('model', RandomForestRegressor(random_state=42))
    ])

    param_dist = {
        'preprocessor__scatter_correction': ['snv', 'msc', None],
        'preprocessor__baseline_correction': ['asls', 'detrend', None],
        'preprocessor__sg_window': [5, 11, 21, 31],
        'preprocessor__sg_poly': [1, 2, 3, 4],
        'preprocessor__sg_deriv': [0, 1, 2],

        'model__n_estimators': [100, 200, 300, 500],
        'model__max_features': ['sqrt', 'log2', 0.8],
        'model__max_depth': [10, 20, 30, None],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],

        'vip__n_components': [5, 10, 15, 20, None],
        'vip__vip_threshold': [0.8, 0.9, 1, 1.1]
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=500,
        cv=inner_cv,
        scoring='r2',
        n_jobs=-1,
        random_state=42
    )

    return search

def xgb_tuning(inner_cv):
    pipeline = Pipeline([
        # ('outlier_remover', OutlierRemover()),
        ('preprocessor', Preprocessor()),
        ('scaler', RobustScaler()),
        ('vip', VIPSelector()),
        ('model', XGBRegressor(random_state=42))
    ])

    param_dist = {
        'preprocessor__scatter_correction': ['snv', 'msc', None],
        'preprocessor__baseline_correction': ['asls', 'detrend', None],
        'preprocessor__sg_window': [5, 11, 21, 31],
        'preprocessor__sg_poly': [1, 2, 3, 4],
        'preprocessor__sg_deriv': [0, 1, 2],

        
        'model__n_estimators': [100, 200, 300, 500],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__max_depth': [3, 5, 7, 9],
        'model__subsample': [0.7, 0.9, 1.0],
        'model__colsample_bytree': [0.7, 0.9, 1.0],
        'model__gamma': [0, 0.1, 0.2],

        'vip__n_components': [5, 10, 15, 20, None],
        'vip__vip_threshold': [0.8, 0.9, 1, 1.1]
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=500,
        cv=inner_cv,
        scoring='r2',
        n_jobs=-1,
        random_state=42
    )

    return search

def remove_y_outliers(df):
    q_low = df['result'].quantile(0.05)
    q_high = df['result'].quantile(0.95)

    df = df[(df['result'] >= q_low) & (df['result'] <= q_high)]
    y = df['result'].copy()
    X = df.drop('result', axis=1)
    plot_Y(y, "y_raw_no_outliers")
    return X, y


if __name__ == '__main__':
        for dataset_name in load_datasets():
            if not dataset_name.startswith("MO_"):
                continue
            window_size = [5, 11, 21, 31]
            poly_order = [1, 2, 3, 4]
            deriv = [0, 1, 2]
            best_r2 = {'params': {}, 'score': -10, "model": None}
            best_rmse = {'params': {}, 'score': 1000, "model": None}
            dataframe = pd.read_csv("./generated_datasets/" + dataset_name, decimal='.')
            print("\n--------------------" + dataset_name + "--------------------\n")
            df = handle_dataset_data(dataframe)
            X_raw, y_raw = remove_y_outliers(df)
            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            # X_train, y_train = remove_outliers(X_train, y_train)
            # print(X_train.shape)
            # plot_row_graph(X_raw, "train_data")
            # plot_row_graph(y_raw, "test_data")
            ## TODO stacking
            ## TODO try RFECV for non-linear models
            outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
            inner_cv = KFold(n_splits=5, shuffle=True, random_state=123)
            pipelines = create_pipelines(inner_cv)
            for pipeline in pipelines:
                rmse, r2 = fit(X_raw, y_raw, pipeline, outer_cv)
            print(f"Best R2 score:  {best_r2}")
            print(f"Best RMSE score:  {best_rmse}")