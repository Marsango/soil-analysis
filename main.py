import numpy as np
import pandas as pd
from pybaselines import Baseline
from scipy.signal import savgol_filter, argrelmin
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


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

def apply_standardization(df):
    scaler = StandardScaler()
    return pd.DataFrame(
        scaler.fit_transform(df),
        index=df.index,
        columns=df.columns
    )

def read_data():
    return pd.read_csv("data_nir_187_samples.csv")

def handle_dataset_data(df):
    df = df.drop(df.columns[0], axis=1)
    return df.replace(',', '.', regex=True).astype(float)

def pre_processing(df):
    df = df.copy()
    df = apply_savgol_to_df(df) # seems good
    # df = apply_baseline_to_df(df) # does not make any sense
    # df = apply_standardization(df) # does not make any sense
    df = apply_snv(df) # change scales
    df = apply_normalization(df) # invert the direction of signal
    return df


def plot_row_graph(original, filtered):
    plt.figure(figsize=(10, 5))
    plt.plot(original, label="Original", alpha=0.7)
    plt.plot(filtered, label="Filtered (Savitzky–Golay)", linewidth=2)
    plt.title("Row 0 - Savitzky–Golay Filter")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig("filtered_row_0.png", dpi=300)

def fit(df, model): ## just a example, needs a update
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    clf = svm.SVC(kernel='linear', C=1.0, random_state=42)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

if __name__ == '__main__':
    df = read_data()
    df = handle_dataset_data(df)
    filtered_df = pre_processing(df)
    original = df.iloc[2].values.astype(float)
    filtered = filtered_df.iloc[2]
    print(filtered)
    plot_row_graph(original, filtered)
