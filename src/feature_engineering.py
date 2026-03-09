"""This module contains functions for feature engineering time series data,
including time delay embedding techniques (lag features and rolling window
features) and temporal embedding techniques (calendar features,
time elapsed features, and Fourier terms). These engineered features can
help capture temporal patterns and improve forecasting model performance."""

import pandas as pd
import numpy as np


def create_lag_features(
    dataset: pd.DataFrame,
    target: str,
    timestamps_per_day: int,
    horizon: int,
) -> pd.DataFrame:
    """Create lag features for 1 day, 1 week, and 1 month for each forecast
    horizon.

    Args:
        dataset: DataFrame with target column
        target: Name of target column
        timestamps_per_day: Number of timestamps per day (24 for hourly)
        horizon: Forecast horizon in timestamps

    Returns:
        DataFrame with lag features added
    """
    data = dataset[[target]].copy()

    # 1 week = 7 days, 1 month = 30 days
    lag_periods = {
        "1_day": timestamps_per_day,
        "1_week": 7 * timestamps_per_day,
        "1_month": 30 * timestamps_per_day,
    }

    for lag_name, lag_value in lag_periods.items():
        col_name = f"lag_{lag_name}_{target}_horizon_{horizon}"
        data[col_name] = dataset[target].shift(
            horizon + (lag_value - timestamps_per_day)
        )
    data.dropna(inplace=True)
    return data


def create_rolling_features(
    dataset: pd.DataFrame,
    target: str,
    timestamps_per_day: int,
    horizon: int,
) -> pd.DataFrame:
    """Create rolling mean, std, min, max features for 1 day, 1 week,
    and 1 month for each forecast horizon.

    Args:
        dataset: DataFrame with target column
        target: Name of target column
        timestamps_per_day: Number of timestamps per day (24 for hourly)
        horizon: Forecast horizon in timestamps

    Returns:
        DataFrame with rolling features
    """
    data = dataset[[target]].copy()

    windows = [timestamps_per_day,
               7 * timestamps_per_day,
               30 * timestamps_per_day]  # 1 day, 1 week, 1 month in hours
    for window in windows:
        # Shift by horizon to ensure we only use data available
        # before forecast time
        data[f"rolling_mean_{window}_horizon_{horizon}"] = (
            dataset[target].shift(horizon).rolling(window=window).mean()
        )
        data[f"rolling_std_{window}_horizon_{horizon}"] = (
            dataset[target].shift(horizon).rolling(window=window).std()
        )
        data[f"rolling_min_{window}_horizon_{horizon}"] = (
            dataset[target].shift(horizon).rolling(window=window).min()
        )
        data[f"rolling_max_{window}_horizon_{horizon}"] = (
            dataset[target].shift(horizon).rolling(window=window).max()
        )
    data.dropna(inplace=True)
    return data


def create_calendar_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Create calendar features such as year, month, hour, day of week,
    is_weekend, and day of year.

    Args:
        dataset: DataFrame with datetime index

    Returns:
        DataFrame with calendar features added
    """
    data = dataset.copy()
    data["year"] = data.index.year
    data["month"] = data.index.month
    data["hour"] = data.index.hour
    data["dayofweek"] = data.index.dayofweek
    data["is_weekend"] = data["dayofweek"].isin([5, 6]).astype(int)
    data["day_of_year"] = data.index.dayofyear
    return data


def create_time_elapsed_feature(dataset: pd.DataFrame) -> pd.DataFrame:
    """Create a continuous time feature representing the amount of
    time elapsed. It converts datetime to Unix timestamp
    (seconds since January 1, 1970):

    Args:
        dataset: DataFrame with datetime index

    Returns:
        DataFrame with 'time_elapsed' feature added
    """
    data = dataset.copy()
    data['time_elapsed'] = data.index.values.astype(np.int64) / (10**9)
    return data


def create_daily_fourier_terms(dataset: pd.DataFrame) -> pd.DataFrame:
    """Create daily Fourier terms to capture daily seasonality.
    A common choice is to use the first 3 pairs of sine and cosine terms.

    Args:
        dataset: DataFrame with datetime index and 'hour' column

    Returns:
        DataFrame with daily Fourier terms added
    """
    data = dataset.copy()
    data["hour"] = data.index.hour
    for k in range(1, 4):
        data[f"hour_sin_{k}"] = np.sin(2 * np.pi * k * data["hour"] / 24)
        data[f"hour_cos_{k}"] = np.cos(2 * np.pi * k * data["hour"] / 24)
    return data


def create_yearly_fourier_terms(dataset: pd.DataFrame) -> pd.DataFrame:
    """Create yearly Fourier terms to capture yearly seasonality.
    A common choice is to use the first 2 pairs of sine and cosine terms.

    Args:
        dataset: DataFrame with datetime index and 'day_of_year' column
    Returns:
        DataFrame with yearly Fourier terms added
    """
    data = dataset.copy()
    data["day_of_year"] = data.index.dayofyear
    for k in range(1, 3):
        data[f"doy_sin_{k}"] = np.sin(2 * np.pi * k * data["day_of_year"] / 365.25)
        data[f"doy_cos_{k}"] = np.cos(2 * np.pi * k * data["day_of_year"] / 365.25)
    return data
