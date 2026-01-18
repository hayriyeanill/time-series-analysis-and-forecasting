"This module contains time series specific analysis functions."

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


def test_stationarity(data: pd.DataFrame, target: str):
    """Perform Augmented Dickey-Fuller test for stationarity.

    Args:
        data (pd.DataFrame): The input DataFrame containing the time series data
        target (str): Column name of the target time series variable

    Returns:
        None: Prints the ADF test results"""

    result = adfuller(data[target])
    labels = [
        "ADF Test Statistic",
        "p-value",
        "#Lags Used",
        "Number of Observations",
    ]

    for value, label in zip(result, labels):
        print(f"{label}: {value}")

    if result[1] <= 0.05:
        print("Data is stationary")
    else:
        print("Data is not stationary")


def apply_differencing(
    data: pd.DataFrame, target: str, order: int = 1
) -> pd.DataFrame:
    """Apply differencing to a time series to achieve stationarity.

    Args:
        data (pd.DataFrame): The input DataFrame containing the time series data
        target (str): Column name of the target time series variable
        order (int): The order of differencing to apply (default is 1)

    Returns:
        pd.DataFrame: A new DataFrame with the differenced time series column added
    """
    differenced_data = data.copy()
    differenced_column_name = f"{target}_diff_{order}"
    differenced_data[differenced_column_name] = differenced_data[target].diff(
        order
    )
    return differenced_data.dropna()


def apply_transformation_log(data: pd.DataFrame, target: str) -> pd.DataFrame:
    """Apply logarithmic transformation to a time series.
    Args:
        data (pd.DataFrame): The input DataFrame containing the time series data
        target (str): Column name of the target time series variable
    Returns:
        pd.DataFrame: A new DataFrame with the log-transformed time series column added
    """
    transformed_data = data.copy()
    transformed_column_name = f"{target}_log"
    transformed_data[transformed_column_name] = np.log(transformed_data[target])
    return transformed_data.dropna()


def apply_transformation_sqrt(data: pd.DataFrame, target: str) -> pd.DataFrame:
    """Apply square root transformation to a time series.
    Args:
        data (pd.DataFrame): The input DataFrame containing the time series data
        target (str): Column name of the target time series variable
    Returns:
        pd.DataFrame: A new DataFrame with the log-transformed time series column added
    """
    transformed_data = data.copy()
    transformed_column_name = f"{target}_sqrt"
    transformed_data[transformed_column_name] = np.sqrt(
        transformed_data[target]
    )
    return transformed_data.dropna()


def apply_seasonal_differencing(
    data: pd.DataFrame, target: str, seasonal_lag: int
) -> pd.DataFrame:
    """Apply seasonal differencing to a time series to achieve stationarity.

    Args:
        data (pd.DataFrame): The input DataFrame containing the time series data
        target (str): Column name of the target time series variable
        seasonal_lag (int): The seasonal lag period for differencing

    Returns:
        pd.DataFrame: A new DataFrame with the seasonally differenced time series column added
    """
    seosanal_differenced_data = data.copy()
    differenced_column_name = f"{target}_seasonal_diff"
    seosanal_differenced_data[
        differenced_column_name
    ] = seosanal_differenced_data[target] - seosanal_differenced_data[
        target
    ].shift(
        seasonal_lag
    )
    return seosanal_differenced_data.dropna()


def plot_autocorrelation_function(data: pd.DataFrame, target: str, lags: int):
    """Apply autocorrelation plot."""
    plt.figure(figsize=(10, 5))
    plot_acf(data[target], lags=lags)
    plt.title("Autocorrelation Function (ACF) of Temperature")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.show()


def plot_partial_correlation_function(
    data: pd.DataFrame, target: str, lags: int
):
    """Apply partial correlation plot."""
    plt.figure(figsize=(10, 5))
    plot_pacf(data[target], lags=lags)
    plt.title("Partial Autocorrelation Function (PACF) of Temperature")
    plt.xlabel("Lag")
    plt.ylabel("Partial Autocorrelation")
    plt.show()
