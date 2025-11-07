
import numpy as np
from sklearn import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
from PreProcessor import Preprocessor
from VIPSelector import VIPSelector
from main import remove_y_outliers, handle_dataset_data, remove_outliers
import pandas as pd

from optuna_all_models import generate_mixup_augmentation

def run_cv_for_trial(pipeline, should_remove_outliers, allow_augmentation, n_new_per_sample, alpha,
                     contamination=0.05):

    kf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

    scores = []

    for step, (train_idx, test_idx) in enumerate(kf.split(X_raw, y_raw)):

        pipeline_clone = clone(pipeline)
        X_train, X_test = X_raw.iloc[train_idx], X_raw.iloc[test_idx]
        y_train, y_test = y_raw.iloc[train_idx], y_raw.iloc[test_idx]

        if should_remove_outliers:
            X_train, y_train = remove_outliers(X_train, y_train, contamination)

        if allow_augmentation:
            X_train, y_train = generate_mixup_augmentation(X_train, y_train, n_new_per_sample, alpha)
        print("Im here")
        pipeline_clone.fit(X_train, y_train)
        y_pred = pipeline_clone.predict(X_test)
        score = - root_mean_squared_error(y_test, y_pred)
        print(score)
        scores.append(score)


    return np.mean(scores)

if __name__ == '__main__':
    dataframe = pd.read_csv("./generated_datasets/MO_187_samples.csv", decimal='.')
    df = handle_dataset_data(dataframe)
    X_raw, y_raw = remove_y_outliers(df)

    estimators = [
        # ('svr', SVR(kernel='linear', C=12.17552345056831, epsilon=0.1)),
        ('plsr', PLSRegression(n_components=5)),
        ('xgb', GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=3, subsample=0.9,
                                          max_features='log2',
                                          random_state=42))
    ]

    final_estimator = RidgeCV()

    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=final_estimator,
        passthrough=False,
        n_jobs=-1
    )

    preprocessing_pipeline = Pipeline([
        ('preprocessor', Preprocessor()),
        ('scaler', RobustScaler()),
        ('vip_selector', VIPSelector())
    ])


    full_pipeline = Pipeline([
        ('preprocessing', preprocessing_pipeline),
        ('model', stacking_regressor)
    ])

    score = run_cv_for_trial(full_pipeline, True, True, 9, 0.4)
    print(f"Stacking RMSE: {score}")