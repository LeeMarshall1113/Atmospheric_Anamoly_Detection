#!/usr/bin/env python
"""
Example end-to-end code using xarray's built-in tutorial dataset "air_temperature"
for demonstration purposes.
"""

import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest

def load_tutorial_data():
    """
    Loads xarray's built-in 'air_temperature' dataset for testing.
    """
    ds = xr.tutorial.open_dataset("air_temperature")
    print("Loaded tutorial dataset:\n", ds)
    return ds

def explore_data(ds):
    """
    Performs basic EDA on a sample location and a single time snapshot.
    Assumes 'air' is a variable in the Dataset with dims (time, lat, lon).
    """
    # 2A. Time-series plot for a specific lat/lon
    sample_point = ds.sel(lat=40.0, lon=-100.0, method="nearest")  # example location
    temp_series = sample_point.air  # variable name is 'air' in this dataset

    plt.figure()
    temp_series.plot()
    plt.title("Air Temperature Time Series at (lat=40.0, lon=-100.0)")
    plt.xlabel("Time")
    plt.ylabel("Temperature (K)")
    plt.show()

    # 2B. Spatial snapshot for the first time index
    temp_snapshot = ds.air.isel(time=0)

    plt.figure()
    temp_snapshot.plot()
    plt.title("Air Temperature Snapshot (time=0)")
    plt.show()

def compute_anomaly(ds):
    """
    Computes a simple anomaly: 'air' minus its mean over time.
    Returns an xarray DataArray called 'temp_anomaly'.
    """
    # climatology_mean along time
    climatology_mean = ds.air.mean(dim="time")
    temp_anomaly = ds.air - climatology_mean
    return temp_anomaly

def run_isolation_forest(ds, var_name="air", contamination=0.01):
    """
    Runs IsolationForest anomaly detection on a single data variable (var_name).
    """
    # Convert the data variable to a DataFrame
    df = ds[var_name].to_dataframe().dropna().reset_index()

    # We'll just use the one variable for demonstration
    X = df[[var_name]].values

    # Train IsolationForest
    iso_forest = IsolationForest(n_estimators=100, 
                                 contamination=contamination, 
                                 random_state=42)
    iso_forest.fit(X)

    # Predict anomalies
    df["anomaly_label"] = iso_forest.predict(X)  # -1 = outlier, +1 = normal

    outliers = df[df["anomaly_label"] == -1]
    print(f"\n--- IsolationForest Results ---")
    print(f"Number of points analyzed: {len(df)}")
    print(f"Number of outliers: {len(outliers)}")
    print("--------------------------------\n")

    # Quick scatter plot (value vs. index)
    plt.figure(figsize=(8,6))
    is_outlier = (df["anomaly_label"] == -1)
    plt.scatter(df.index, df[var_name], c=is_outlier, cmap="coolwarm", alpha=0.3)
    plt.xlabel("Sample Index")
    plt.ylabel(f"{var_name} Value")
    plt.title("Isolation Forest Anomaly Detection")
    plt.show()

    return df

def main():
    # 1. Load tutorial data
    ds = load_tutorial_data()

    # 2. Basic EDA
    explore_data(ds)

    # 3. Simple anomaly measure
    temp_anomaly = compute_anomaly(ds)
    print("temp_anomaly DataArray created with shape:", temp_anomaly.shape)
    ds["temp_anomaly"] = temp_anomaly  # store in dataset

    # 3.1. Run Isolation Forest
    results_df = run_isolation_forest(ds, var_name="air", contamination=0.01)

if __name__ == "__main__":
    main()
