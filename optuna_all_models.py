import traceback
import time
import optuna
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.base import clone
from xgboost import XGBRegressor

from PreProcessor import Preprocessor
from VIPSelector import VIPSelector
import pandas as pd

from main import handle_dataset_data, remove_y_outliers, remove_outliers


def create_common_pipeline_steps(trial):
    scatter_corr = trial.suggest_categorical('scatter_correction', ['snv', 'msc', None])
    baseline_corr = trial.suggest_categorical('baseline_correction', ['asls', 'detrend', None])

    sg_window = trial.suggest_categorical('sg_window', [5, 11, 21, 31])
    sg_poly = trial.suggest_int('sg_poly', 1, 4)
    sg_deriv = trial.suggest_int('sg_deriv', 0, 2)
    sg_enabled = trial.suggest_categorical('sg_enable', [True, False])

    if sg_poly <= sg_deriv:
        raise optuna.exceptions.TrialPruned()

    vip_comp = trial.suggest_categorical('vip_n_components', [5, 10, 15, 20, None])
    vip_thresh = trial.suggest_float('vip_vip_threshold', 0.8, 1.1)

    preprocessor = Preprocessor(
        scatter_correction=scatter_corr,
        baseline_correction=baseline_corr,
        sg_window=sg_window,
        sg_poly=sg_poly,
        sg_deriv=sg_deriv,
        sg_enabled=sg_enabled
    )

    scaler = RobustScaler()

    vip = VIPSelector(
        n_components=vip_comp,
        vip_threshold=vip_thresh
    )

    return [
        ('preprocessor', preprocessor),
        ('scaler', scaler),
        ('vip', vip)
    ]


def run_cv_for_trial(pipeline, trial, should_remove_outliers):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = []

    for step, (train_idx, test_idx) in enumerate(kf.split(X_raw, y_raw)):

        pipeline_clone = clone(pipeline)
        X_train, X_test = X_raw.iloc[train_idx], X_raw.iloc[test_idx]
        y_train, y_test = y_raw.iloc[train_idx], y_raw.iloc[test_idx]

        if should_remove_outliers:
            X_train, y_train = remove_outliers(X_train, y_train)

        try:
            pipeline_clone.fit(X_train, y_train)
            score = pipeline_clone.score(X_test, y_test)
            scores.append(score)

            trial.report(score, step)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        except Exception as e:
            print(f"Fold failed with error: {e}")
            traceback.print_exc()
            trial.report(-1.0, step)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            return -1.0

    return np.mean(scores)


def objective_plsr(trial):
    pipeline_steps = create_common_pipeline_steps(trial)

    n_components = trial.suggest_categorical('n_components', [5, 10, 15, 20])
    should_remove_outliers = trial.suggest_categorical('remove_outliers', [True, False])

    model = PLSRegression(n_components=n_components)
    pipeline_steps.append(('model', model))

    pipeline = Pipeline(pipeline_steps)
    return run_cv_for_trial(pipeline, trial, should_remove_outliers)


def objective_svr(trial):
    pipeline_steps = create_common_pipeline_steps(trial)

    kernel = trial.suggest_categorical('kernel', ['rbf', 'linear'])
    should_remove_outliers = trial.suggest_categorical('remove_outliers', [True, False])

    model_params = {'kernel': kernel}

    if kernel == 'rbf':
        model_params['C'] = trial.suggest_float('C', 1e-1, 1e3, log=True)
        model_params['gamma'] = trial.suggest_float('gamma', 1e-4, 1.0, log=True)

    elif kernel == 'linear':
        model_params['C'] = trial.suggest_float('C', 1e-3, 1e2, log=True)

    model = SVR(**model_params)
    pipeline_steps.append(('model', model))

    pipeline = Pipeline(pipeline_steps)
    return run_cv_for_trial(pipeline, trial, should_remove_outliers)


def objective_gbr(trial):
    pipeline_steps = create_common_pipeline_steps(trial)

    should_remove_outliers = trial.suggest_categorical('remove_outliers', [True, False])
    n_estimators = trial.suggest_categorical('n_estimators', [100, 200, 300, 500])
    learning_rate = trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.1])
    max_depth = trial.suggest_categorical('max_depth', [3, 5, 8, 10])
    subsample = trial.suggest_categorical('subsample', [0.7, 0.9, 1.0])
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])

    model = GradientBoostingRegressor(
        random_state=42,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        max_features=max_features
    )
    pipeline_steps.append(('model', model))

    pipeline = Pipeline(pipeline_steps)
    return run_cv_for_trial(pipeline, trial, should_remove_outliers)


def objective_rf(trial):
    pipeline_steps = create_common_pipeline_steps(trial)

    should_remove_outliers = trial.suggest_categorical('remove_outliers', [True, False])
    n_estimators = trial.suggest_categorical('n_estimators', [100, 200, 300, 500])
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.8])
    max_depth = trial.suggest_categorical('max_depth', [10, 20, 30, None])
    min_samples_split = trial.suggest_categorical('min_samples_split', [2, 5, 10])
    min_samples_leaf = trial.suggest_categorical('min_samples_leaf', [1, 2, 4])

    model = RandomForestRegressor(
        random_state=42,
        n_jobs=1,
        n_estimators=n_estimators,
        max_features=max_features,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )
    pipeline_steps.append(('model', model))

    pipeline = Pipeline(pipeline_steps)
    return run_cv_for_trial(pipeline, trial, should_remove_outliers)


def objective_xgb(trial):
    pipeline_steps = create_common_pipeline_steps(trial)

    should_remove_outliers = trial.suggest_categorical('remove_outliers', [True, False])
    n_estimators = trial.suggest_categorical('n_estimators', [100, 200, 300, 500])
    learning_rate = trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.1])
    max_depth = trial.suggest_categorical('max_depth', [3, 5, 7, 9])
    subsample = trial.suggest_categorical('subsample', [0.7, 0.9, 1.0])
    colsample_bytree = trial.suggest_categorical('colsample_bytree', [0.7, 0.9, 1.0])
    gamma = trial.suggest_categorical('gamma', [0, 0.1, 0.2])

    model = XGBRegressor(
        random_state=42,
        n_jobs=1,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma
    )
    pipeline_steps.append(('model', model))

    pipeline = Pipeline(pipeline_steps)
    return run_cv_for_trial(pipeline, trial, should_remove_outliers)


def run_all_model_studies(n_trials_per_model=100):
    model_objectives = {
        "PLSR": objective_plsr,
        "SVR": objective_svr,
        "GradientBoosting": objective_gbr,
        "RandomForest": objective_rf,
        "XGBoost": objective_xgb
    }

    all_studies = {}

    for model_name, objective_func in model_objectives.items():
        print(f"Running study for {model_name}")
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)

        study = optuna.create_study(direction='maximize', pruner=pruner)
        study.optimize(objective_func, n_trials=n_trials_per_model, show_progress_bar=True)
        all_studies[model_name] = study

        print(f"Best {model_name}  r²: {study.best_value:.4f}\n")

    print("Best r² scores:")
    for model_name, study in all_studies.items():
        print(f"  {model_name}: {study.best_value:.4f}")

    return all_studies


if __name__ == "__main__":
    start = time.time()
    dataframe = pd.read_csv("./generated_datasets/MO_187_samples.csv", decimal='.')
    df = handle_dataset_data(dataframe)
    X_raw, y_raw = remove_y_outliers(df)
    completed_studies = run_all_model_studies(n_trials_per_model=2000)
    end = time.time()
    elapsed = end - start
    print(f"Time elapsed: {elapsed}s")
