import os

import numpy as np
import pandas as pd
from pybaselines import Baseline
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def apply_savgol_to_df(df):
    data = df.to_numpy(dtype=float)
    filtered = np.apply_along_axis(lambda x: savgol_filter(x, 11, 3), 1, data)
    return pd.DataFrame(filtered, columns=df.columns, index=df.index)

def apply_baseline_to_df(df):
    baseline_fitter = Baseline()
    data = df.to_numpy(dtype=float)

    corrected = np.zeros_like(data)
    for i, row in enumerate(data):
        baseline, _ = baseline_fitter.asls(row, lam=1e6, p=0.01)
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

## Unused because its already done in the pipeline with the StandardScaler
def apply_standardization(df):
    scaler = StandardScaler()
    return pd.DataFrame(
        scaler.fit_transform(df),
        index=df.index,
        columns=df.columns
    )

def handle_dataset_data(df):
    df = df.drop(df.columns[0], axis=1)
    return df.replace(',', '.', regex=True).astype(float)

def pre_processing(df):
    df = df.copy()
    df = apply_savgol_to_df(df) # seems good
    # df = apply_baseline_to_df(df) # does not make any sense
    # df = apply_standardization(df) # does not make any sense
    df = apply_snv(df) # change scales
    # df = apply_normalization(df) # invert the direction of signal
    return df


def plot_row_graph(original, filtered, dataset_name):
    plt.figure(figsize=(10, 5))
    plt.plot(original, label="Original", alpha=0.7)
    plt.plot(filtered, label="Filtered (Savitzky–Golay)", linewidth=2)
    plt.title("Row 0 - Savitzky–Golay Filter")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(f"filtered_{dataset_name}.png", dpi=300)

def split_data_train_test(df):
    X = df.drop("result", axis=1)
    y = df["result"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def fit(df, pipeline):
    model = pipeline[0]
    model_name = pipeline[1]
    X_train, X_test, y_train, y_test = split_data_train_test(df)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{model_name} | MSE: {root_mean_squared_error(y_test, y_pred)} | R²: {r2_score(y_test, y_pred)}")

def load_datasets():
    files = os.listdir("generated_datasets")
    return [f for f in files if "187_samples" in f]

def create_pipelines(df):
    pipeline_list = []
    pipeline_list.append((svr_tunning(df), "SVR"))
    pipeline_list.append((Pipeline([
        ('scaler', StandardScaler()),
        ('plsr', PLSRegression())
    ]), "PLSR"))
    pipeline_list.append((random_forest_tuning(df), "RandomForest"))
    return pipeline_list

def svr_tunning(df):
    X_train, X_test, y_train, y_test = split_data_train_test(df)
    param1 = {
        'svr__kernel': ['poly'],
        'svr__C': [1, 5],
        'svr__degree': [3, 5],
        'svr__gamma': ['auto', 'scale'],
        'svr__coef0': [0.01, 10, 0.5]
    }
    param2 = {
        'svr__kernel': ['linear', 'rbf', 'sigmoid'],
        'svr__C': [1, 5, 10],
        'svr__gamma': ['auto', 'scale']
    }
    grid_search1 = GridSearchCV(make_pipeline(StandardScaler(), SVR()), param1, cv=5, n_jobs=-1)
    grid_search2 = GridSearchCV(make_pipeline(StandardScaler(), SVR()), param2, cv=5, n_jobs=-1)
    grid_search1.fit(X_train, y_train)
    grid_search2.fit(X_train, y_train)
    best_pipeline1 = grid_search1.best_estimator_
    best_pipeline2 = grid_search2.best_estimator_
    y_pred_best1 = best_pipeline1.predict(X_test)
    y_pred_best2 = best_pipeline2.predict(X_test)
    accuracy_best1 = r2_score(y_test, y_pred_best1)
    accuracy_best2 = r2_score(y_test, y_pred_best2)
    return grid_search1.best_estimator_ if accuracy_best1 > accuracy_best2 else grid_search2.best_estimator_

def random_forest_tuning(df):
    X_train, X_test, y_train, y_test = split_data_train_test(df)
    param_dist = {
        'n_estimators': [100, 200],
        'max_features': ['sqrt', 0.8],
        'max_depth': [10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    search = RandomizedSearchCV(
        estimator=RandomForestRegressor(),
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        random_state=42
    )
    search.fit(X_train, y_train)
    return search.best_estimator_

if __name__ == '__main__':
        for dataset_name in load_datasets():
            print("\n--------------------" + dataset_name + "--------------------\n")
            dataframe = pd.read_csv("./generated_datasets/" + dataset_name)
            df = handle_dataset_data(dataframe)
            filtered_df = pre_processing(df)
            # original = df.iloc[2].values.astype(float)
            # filtered = filtered_df.iloc[2]
            pipelines = create_pipelines(filtered_df)
            for pipeline in pipelines:
                fit(filtered_df, pipeline)
            # plot_row_graph(original, filtered, dataset_name)
