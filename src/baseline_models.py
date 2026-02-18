"""Baseline forecasting models for time series analysis.

This module provides simple baseline models including arithmetic mean,
rolling mean, and seasonal naive approaches for both single-step and
multi-horizon forecasting.
"""


import pandas as pd
import numpy as np


def baseline_arithmetic_mean_model(
    train_set: pd.DataFrame,
    test_set: pd.DataFrame,
    target: str,
    window_size: int,
) -> pd.DataFrame:
    """Arithmetic mean baseline model. There exists a single constant level.
    Compute one mean from history. Predict this constant mean for all future
    timestamps.

    Args:
        train_set (pd.DataFrame): Training dataset
        test_set (pd.DataFrame): Test dataset
        target (str): Target column name
        window_size (int): Number of past observations to use for the mean

    Returns:
        pd.DataFrame: DataFrame with actual and predicted values
    """
    # compute historical mean from the last `window_size` of training
    mean_temp = train_set.tail(window_size).mean()
    # predict a constant mean for all test timestamps
    mean_pred = np.full(shape=len(test_set), fill_value=mean_temp)
    arithmetic_mean_baseline_df = pd.DataFrame()
    arithmetic_mean_baseline_df[target] = test_set[target]
    arithmetic_mean_baseline_df["arithmetic_mean_pred"] = mean_pred
    return arithmetic_mean_baseline_df


def baseline_rolling_mean_model(
    data: pd.DataFrame, test_set: pd.DataFrame, target: str, window_size: int
) -> pd.DataFrame:
    """Rolling mean baseline model.The level slowly changes over time.
    At each time step, recompute the mean of the past `window_size`
    observations.

    Args:
        data (pd.DataFrame): Full dataset (train + test)
        target (str): Target column name
        window_size (int): Number of past observations to use for the rolling mean
    Returns:
        pd.DataFrame: DataFrame with actual and predicted values
    """
    # prediction at time t uses mean of [t-window, ..., t-1]
    pred_all = data.shift(1).rolling(window_size).mean()

    # keep only test part and drop early NaNs
    y_true = test_set[target]
    y_pred = pred_all.loc[y_true.index].dropna()
    y_true = y_true.loc[y_pred.index]

    rolling_mean_baseline_df = pd.DataFrame()
    rolling_mean_baseline_df[target] = y_true
    rolling_mean_baseline_df["rolling_mean_pred"] = y_pred
    return rolling_mean_baseline_df


def baseline_model_seasonal_naive(
    test_set: pd.DataFrame, target: str, period: int
) -> pd.DataFrame:
    """Create a seasonal naive persistence model with a specified period.

    Args:
        test_set (pd.DataFrame): Test dataset
        target (str): Target column name
        period (int): Seasonal period (in number of time steps)

    Returns:
        pd.DataFrame: DataFrame with actual and predicted values
    """
    baseline_df = pd.DataFrame()
    shifted_column = test_set[target].shift(period)
    baseline_df[target] = test_set[target]
    baseline_df[f"H+{period}"] = shifted_column
    baseline_df.dropna(inplace=True)
    return baseline_df


def baseline_arithmetic_mean_for_multihorizon(
    train_set: pd.DataFrame,
    test_set: pd.DataFrame,
    target: str,
    window_size: int,
    forecast_days: int,
    timestamps_per_day: int,
) -> pd.DataFrame:
    """
    Arithmetic mean baseline model for multi-horizon forecasting.

    Computes a single arithmetic mean from the last window_size observations
    in the training set and uses this constant value as prediction for all
    forecast horizons.

    Args:
        train_set (pd.DataFrame): Training dataset
        test_set (pd.DataFrame): Test dataset
        target (str): Target column name
        window_size (int): Number of past observations to use for the mean
        forecast_days (int): Number of days to forecast
        timestamps_per_day (int): Number of timestamps per day

    Returns:
        DataFrame with actual values and predictions for each horizon
        (H+24, H+48, etc.)
    """
    arithmetic_mean_baseline_df = pd.DataFrame()
    arithmetic_mean_baseline_df[target] = test_set[target]

    for i in range(1, forecast_days + 1):
        period = i * timestamps_per_day
        mean_temp = train_set.tail(window_size).mean()
        mean_pred = np.full(shape=len(test_set), fill_value=mean_temp)
        arithmetic_mean_baseline_df[target] = test_set[target]
        arithmetic_mean_baseline_df[f"H+{period}"] = mean_pred
    return arithmetic_mean_baseline_df


def baseline_rolling_mean_for_multihorizon(
    data: pd.DataFrame,
    test_set: pd.DataFrame,
    target: str,
    window_size: int,
    forecast_days: int,
    timestamps_per_day: int,
) -> pd.DataFrame:
    """Rolling mean baseline model for multi-horizon forecasting.

    Computes a rolling mean of the past window_size observations (using shift(1)
    to prevent data leakage) at each time step. The level slowly adapts over time
    as new observations become available. The rolling mean represents the forecast level
    available at time t.

    Args:
        data: Full dataset (train + test)
        target: Target column name
        window_size: Number of past observations for rolling mean calculation
        forecast_days: Number of days to forecast
        timestamps_per_day: Number of timestamps per day (e.g., 24 for hourly)

    Returns:
        DataFrame with actual values and predictions for each horizon
        (H+24, H+48, etc.)
    """
    rolling_mean_baseline_df = pd.DataFrame()

    # Rolling arithmetic mean (no leakage)
    rolling_mean = data.shift(1).rolling(window_size).mean()
    rolling_mean_baseline_df[target] = test_set[target]

    for i in range(1, forecast_days + 1):
        period = i * timestamps_per_day
        rolling_mean_baseline_df[f"H+{period}"] = rolling_mean.shift(-period)
    rolling_mean_baseline_df.dropna(inplace=True)
    return rolling_mean_baseline_df


def baseline_model_seasonal_naive_1h_for_multihorizon(
    test_set: pd.DataFrame,
    target: str,
    forecast_days: int,
    timestamps_per_day: int,
) -> pd.DataFrame:
    """Hourly persistence baseline model for multi-horizon forecasting.

    Uses the value from 1 hour ago as prediction for each forecast horizon.
    This is the simplest form of persistence model where each prediction
    is based on the observation from exactly 1 time step back.

    Args:
        test_set: Test dataset
        target: Target column name
        forecast_days: Number of days to forecast
        timestamps_per_day: Number of timestamps per day (e.g., 24 for hourly)

    Returns:
        DataFrame with actual values and predictions for each horizon
        (H+24, H+48, etc.)
    """
    baseline_df = pd.DataFrame()
    baseline_df[target] = test_set[target]
    lag1 = test_set[target].shift(1)
    for i in range(1, forecast_days + 1):
        period = i * timestamps_per_day
        baseline_df[f"H+{period}"] = lag1.shift(period)

    baseline_df.dropna(inplace=True)
    return baseline_df


def baseline_model_seasonal_naive_24h_for_multihorizon(
    test_set: pd.DataFrame,
    target: str,
    forecast_days: int,
    timestamps_per_day: int,
) -> pd.DataFrame:
    """Daily seasonal naive baseline model for multi-horizon forecasting.

    Uses the value from exactly 24 hours ago (same time yesterday) as
    prediction for each forecast horizon. This captures daily seasonality
    patterns by assuming that values repeat with a 24-hour cycle.

    Args:
        test_set: Test dataset
        target: Target column name
        forecast_days: Number of days to forecast
        timestamps_per_day: Number of timestamps per day (e.g., 24 for hourly)

    Returns:
        DataFrame with actual values and predictions for each horizon
        (H+24, H+48, etc.)
    """
    baseline_df = pd.DataFrame()
    baseline_df[target] = test_set[target]
    for i in range(1, forecast_days + 1):
        period = i * timestamps_per_day
        shifted_column = test_set[target].shift(period)
        baseline_df[f"H+{period}"] = shifted_column

    baseline_df.dropna(inplace=True)
    return baseline_df
