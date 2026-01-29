"""KPIs for time series forecasting evaluation.

This module provides functions to calculate various error metrics including
MAE, RMSE, Bias, and their normalized versions for model performance
assessment.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def metrics_summary(dataframe: pd.DataFrame, actual: str, pred: str) -> dict:
    """Calculate base and normalized error metrics.

    Args:
        dataframe (pd.DataFrame): DataFrame containing actual and predicted values
        actual (str): Column name for actual values
        pred (str): Column name for predicted values

    Returns:
        pd.DataFrame: DataFrame containing all calculated metrics

    """
    y_true = dataframe[actual].values
    y_pred = dataframe[pred].values

    # --- Base metrics ---
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = np.mean(y_pred - y_true)

    # --- Normalization (by range) ---
    value_range = y_true.max() - y_true.min()

    nmae = mae / value_range
    nrmse = rmse / value_range

    results = {
        "MAE (°C)": round(mae, 2),
        "RMSE (°C)": round(rmse, 2),
        "Bias (°C)": round(bias, 2),
        "NMAE": round(nmae, 3),
        "NRMSE": round(nrmse, 3)
    }

    return pd.DataFrame([results])
