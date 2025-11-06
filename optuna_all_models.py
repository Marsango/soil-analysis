import traceback
import time
import optuna
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.base import clone
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

from PreProcessor import Preprocessor
from VIPSelector import VIPSelector
import pandas as pd

from main import handle_dataset_data, remove_y_outliers, remove_outliers


def create_common_pipeline_steps(trial):
    scatter_corr = trial.suggest_categorical('scatter_correction', ['snv', 'msc', None])
    baseline_or_cr = trial.suggest_categorical('baseline_mode', ['baseline', 'cr', None])

    baseline_corr = None
    continuum_rem = False

    if baseline_or_cr == 'baseline':
        baseline_corr = trial.suggest_categorical('baseline_correction', ['asls', 'detrend', None])
    elif baseline_or_cr == 'cr':
        continuum_rem = True


    sg_window = trial.suggest_categorical('sg_window', [5, 11, 21, 31])
    sg_poly = trial.suggest_int('sg_poly', 1, 4)
    sg_deriv = trial.suggest_int('sg_deriv', 0, 2)
    sg_enabled = trial.suggest_categorical('sg_enable', [True, False])

    if sg_poly < sg_deriv:
        raise optuna.exceptions.TrialPruned()

    vip_comp = trial.suggest_categorical('vip_n_components', [5, 10, 15, 20, None])
    vip_thresh = trial.suggest_float('vip_vip_threshold', 0.8, 1.1)

    preprocessor = Preprocessor(
        scatter_correction=scatter_corr,
        baseline_correction=baseline_corr,
        sg_window=sg_window,
        sg_poly=sg_poly,
        sg_deriv=sg_deriv,
        sg_enabled=sg_enabled,
        continuum_removal=continuum_rem
    )

    scaler = RobustScaler()

    vip = VIPSelector(
        n_components=vip_comp,
        vip_threshold=vip_thresh
    )

    raw_pipeline = [('preprocessor', preprocessor)]
    enable_scaler = trial.suggest_categorical('enable_scaler', [True, False])
    if enable_scaler:
        raw_pipeline.append(('scaler', scaler))

    raw_pipeline.append(('vip', vip))

    return raw_pipeline

def plot_test_train_data(pipeline, X_test, y_test, fold):
    y_pred = pipeline.predict(X_test)

    plot_df = pd.DataFrame({
        'actual': y_test.reset_index(drop=True),
        'predicted': y_pred
    })

    plot_df_sorted = plot_df.sort_values(by='actual').reset_index(drop=True)

    plt.figure(figsize=(12, 7))

    plt.scatter(plot_df_sorted.index, plot_df_sorted['actual'],
                label='Actual Test Data (Points)', color='blue', alpha=0.7, s=50)
    plt.plot(plot_df_sorted.index, plot_df_sorted['predicted'],
             label='Predicted Values (Line+Points)', color='red', marker='o',
             linestyle='--', markersize=5)

    plt.title('Actual vs. Predicted Values (Sorted by Actual)')
    plt.xlabel('Sample Index (Sorted by Actual Value)')
    plt.ylabel('Y-Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./plots/actual_vs_predicted_sorted_{pipeline.named_steps['model'].__class__.__name__}_{fold}.png')

## based in this article https://arxiv.org/pdf/1710.01927
def generate_data_augmentation(X_train, y_train):
    new_X_train = []
    new_Y_train = []
    std = X_train.values.std()
    original_columns = X_train.columns
    for i, row in X_train.iterrows():
        sample = row.values
        y_val = y_train.loc[i]

        new_X_train.append(sample)
        new_Y_train.append(y_val)
        for j in range(9):
            offset = std * np.random.uniform(low=-0.1, high=0.1)
            multiplier = 1 + (np.random.uniform(low=-0.1, high=0.1) * std)
            slope_vector  = np.linspace(np.random.uniform(0.95, 1.05),
                                                  np.random.uniform(0.95, 1.05), len(sample))
            new_sample = (sample * multiplier * slope_vector) + offset
            new_X_train.append(new_sample)
            new_Y_train.append(y_val)

    new_X_train = pd.DataFrame(new_X_train, columns=original_columns)
    new_Y_train = pd.Series(new_Y_train, name='result')
    return new_X_train, new_Y_train


def generate_mixup_augmentation(X_train, y_train, n_new_per_sample=9, alpha=0.4):
    original_columns = X_train.columns
    X_orig = X_train.values
    y_orig = y_train.values
    n_samples = X_orig.shape[0]

    n_new_samples = n_samples * n_new_per_sample

    sample_a_indices = np.random.randint(0, n_samples, n_new_samples)
    sample_b_indices = np.random.randint(0, n_samples, n_new_samples)

    lmbda = np.random.beta(alpha, alpha, n_new_samples).reshape(-1, 1)

    X_aug = lmbda * X_orig[sample_a_indices] + (1 - lmbda) * X_orig[sample_b_indices]
    y_aug = lmbda.flatten() * y_orig[sample_a_indices] + (1 - lmbda.flatten()) * y_orig[sample_b_indices]

    X_new = np.vstack((X_orig, X_aug))
    y_new = np.concatenate((y_orig, y_aug))

    return pd.DataFrame(X_new, columns=original_columns), pd.Series(y_new, name='result')


def run_cv_for_trial(pipeline, trial, should_remove_outliers, contamination):
    kf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)

    scores = []

    for step, (train_idx, test_idx) in enumerate(kf.split(X_raw, y_raw)):

        pipeline_clone = clone(pipeline)
        X_train, X_test = X_raw.iloc[train_idx], X_raw.iloc[test_idx]
        y_train, y_test = y_raw.iloc[train_idx], y_raw.iloc[test_idx]

        if should_remove_outliers:
            X_train, y_train = remove_outliers(X_train, y_train, contamination)

        X_train, y_train = generate_mixup_augmentation(X_train, y_train)
        try:
            pipeline_clone.fit(X_train, y_train)
            # plot_test_train_data(pipeline_clone, X_test, y_test, step)
            # score = pipeline_clone.score(X_test, y_test)
            y_pred = pipeline_clone.predict(X_test)
            score = - root_mean_squared_error(y_test, y_pred)
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
    contamination = 0.05
    if should_remove_outliers:
        contamination = trial.suggest_float('contamination', 0.01, 0.2)
    model = PLSRegression(n_components=n_components)
    pipeline_steps.append(('model', model))

    pipeline = Pipeline(pipeline_steps)


    return run_cv_for_trial(pipeline, trial, should_remove_outliers, contamination)


def objective_svr(trial):
    pipeline_steps = create_common_pipeline_steps(trial)

    kernel = trial.suggest_categorical('kernel', ['rbf', 'linear'])
    should_remove_outliers = trial.suggest_categorical('remove_outliers', [True, False])
    contamination = 0.05
    if should_remove_outliers:
        contamination = trial.suggest_float('contamination', 0.01, 0.2)
    model_params = {'kernel': kernel}

    if kernel == 'rbf':
        model_params['C'] = trial.suggest_float('C', 1e-1, 1e3, log=True)
        model_params['gamma'] = trial.suggest_float('gamma', 1e-4, 1.0, log=True)

    elif kernel == 'linear':
        model_params['C'] = trial.suggest_float('C', 1e-3, 1e2, log=True)

    model = SVR(**model_params)
    pipeline_steps.append(('model', model))


    pipeline = Pipeline(pipeline_steps)
    return run_cv_for_trial(pipeline, trial, should_remove_outliers, contamination)


def objective_gbr(trial):
    pipeline_steps = create_common_pipeline_steps(trial)

    should_remove_outliers = trial.suggest_categorical('remove_outliers', [True, False])
    contamination = 0.05
    if should_remove_outliers:
        contamination = trial.suggest_float('contamination', 0.01, 0.2)

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
    return run_cv_for_trial(pipeline, trial, should_remove_outliers, contamination)


def objective_rf(trial):
    pipeline_steps = create_common_pipeline_steps(trial)

    should_remove_outliers = trial.suggest_categorical('remove_outliers', [True, False])
    contamination = 0.05
    if should_remove_outliers:
        contamination = trial.suggest_float('contamination', 0.01, 0.2)
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
    return run_cv_for_trial(pipeline, trial, should_remove_outliers, contamination)


def objective_xgb(trial):
    pipeline_steps = create_common_pipeline_steps(trial)

    should_remove_outliers = trial.suggest_categorical('remove_outliers', [True, False])
    contamination = 0.05
    if should_remove_outliers:
        contamination = trial.suggest_float('contamination', 0.01, 0.2)
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
    return run_cv_for_trial(pipeline, trial, should_remove_outliers, contamination)


def run_all_model_studies(n_trials_per_model=100):
    model_objectives = {
        "PLSR": {"function": objective_plsr, "best_knowed_parameters": {'scatter_correction': None, 'baseline_correction': None, 'sg_window': 31, 'sg_poly': 2, 'sg_deriv': 1, 'sg_enable': False, 'vip_n_components': None,
                              'vip_vip_threshold': 0.8725998180281626, 'n_components': 5, 'remove_outliers': False,
                              'contamination': 0.05, 'baseline_mode': None, 'enable_scaler': True}
                 },
        "SVR": {"function": objective_svr, "best_knowed_parameters": {'scatter_correction': None, 'baseline_correction': None, 'sg_window': 5, 'sg_poly': 3, 'sg_deriv': 1,
     'sg_enable': True, 'vip_n_components': 10, 'vip_vip_threshold': 1.0005199201746844, 'kernel': 'rbf',
     'remove_outliers': True, 'C': 784.0639137178465, 'gamma': 0.00010631666741666417,
                              'contamination': 0.05, 'baseline_mode': None, 'enable_scaler': True}
                },
        "GradientBoosting": {"function": objective_gbr, "best_knowed_parameters": {'scatter_correction': None, 'baseline_correction': None, 'sg_window': 11,
                              'sg_poly': 3, 'sg_deriv': 1, 'sg_enable': True, 'vip_n_components': 20,
                              'vip_vip_threshold': 0.9348559509211628, 'remove_outliers': True,
                              'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 3,
                              'subsample': 0.9, 'max_features': 'log2', 'contamination': 0.05,
                              'baseline_mode': None, 'enable_scaler': True}
                             },
        "RandomForest": {"function": objective_rf, "best_knowed_parameters": {'scatter_correction': None, 'baseline_correction': None, 'sg_window': 11,
                              'sg_poly': 4, 'sg_deriv': 1, 'sg_enable': True, 'vip_n_components': 10,
                              'vip_vip_threshold': 0.9241278894976066, 'remove_outliers': False,
                              'n_estimators': 100, 'max_features': 'log2', 'max_depth': 10,
                              'min_samples_split': 5, 'min_samples_leaf': 1, 'contamination': 0.05,
                              'baseline_mode': None, 'enable_scaler': True}
                         },
        "XGBoost": {"function": objective_xgb, "best_knowed_parameters": {'scatter_correction': None, 'baseline_correction': None, 'sg_window': 5, 'sg_poly': 2,
                              'sg_deriv': 1, 'sg_enable': True, 'vip_n_components': 10,
                              'vip_vip_threshold': 0.9336065598099277, 'remove_outliers': False,
                               'n_estimators': 300, 'learning_rate': 0.1, 'max_depth': 5, 'subsample': 0.7,
                              'colsample_bytree': 0.7, 'gamma': 0.2, 'contamination': 0.05,
                              'baseline_mode': None, 'enable_scaler': True}},
    }

    all_studies = {}

    for model_name, objective_func in model_objectives.items():
        print(f"Running study for {model_name}")
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)

        study = optuna.create_study(direction='maximize', pruner=pruner)
        study.enqueue_trial(objective_func["best_knowed_parameters"])
        study.optimize(objective_func["function"], n_trials=n_trials_per_model, show_progress_bar=True, n_jobs=-1)
        all_studies[model_name] = study

        print(f"Best {model_name}  RMSE: {study.best_value:.4f}\n")

    print("Best RMSE scores:")
    for model_name, study in all_studies.items():
        print(f"  {model_name}: {study.best_value:.4f}")
        print(f"    Best Params: {study.best_params}")
    return all_studies


if __name__ == "__main__":
    start = time.time()
    dataframe = pd.read_csv("./generated_datasets/MO_187_samples.csv", decimal='.')
    df = handle_dataset_data(dataframe)
    X_raw, y_raw = remove_y_outliers(df)

    completed_studies = run_all_model_studies(n_trials_per_model=1)
    end = time.time()
    elapsed = end - start
    print(f"Time elapsed: {elapsed}s")

    #plot my predicted data vs real data
    #rfecv
    #data augmentation
    #try both datasets again
    #spend more time in pH
    ##take a look if the parameters of vip are impacting and if they arent cache the results
    ##allow mix up data augmentation to be tuned
    ##try pca
    ##stacking plsr/xgboost
    ##try wavelet denoising
    ##remove some not really useful parameters of the pipeline to find good models faster. (vip,
    # remove baseline correction, remove sg tuning, contamination)
    # try drop wavelengths at end/start again
