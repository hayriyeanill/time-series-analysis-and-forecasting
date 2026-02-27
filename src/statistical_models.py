"""Statistical models for time series forecasting."""
from typing import Tuple
import numpy as np
import pandas as pd

from pmdarima import auto_arima
from pmdarima.arima import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR


def arima_fit_model(data: pd.DataFrame, order: tuple) -> ARIMA:
    """Fit the ARIMA model with order.

    Args:
        data (pd.DataFrame): The input data for fitting the model.
        order (tuple): The (p, d, q) order of the ARIMA model.

    Returns:
        model (ARIMA): The fitted ARIMA model.
    """
    np.random.seed(42)
    model = ARIMA(order=order)
    model.fit(data)
    return model


def arima_predict(model: ARIMA, n_periods: int) -> np.ndarray:
    """Predict the ARIMA model.

    Args:
        model (ARIMA): The fitted ARIMA model.
        n_periods (int): The number of periods to predict.

    Returns:
        np.ndarray: The predicted values.
    """
    return model.predict(n_periods)


def run_arima_model(train: pd.DataFrame,
                    test: pd.DataFrame,
                    target: str,
                    order: tuple) -> Tuple[float, pd.DataFrame]:
    """Run arima fit and predict functions.

    Args:
        train (pd.DataFrame): The training data.
        test (pd.DataFrame): The testing data.
        target (str): The target column name in the DataFrame.
        order (tuple): The (p, d, q) order of the ARIMA model

    Returns:
        pred_frame (pd.DataFrame): The predicted values as a DataFrame.
    """
    model = arima_fit_model(train[target], order)
    pred = arima_predict(model, test.shape[0])
    pred_frame = pd.DataFrame(pred, columns=["prediction"])
    return pred_frame


def autoarima_fit_model(data: pd.DataFrame) -> auto_arima:
    """Fit auto ARIMA model.

    Parameters:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        model (pmdarima.auto_arima): The fitted ARIMA model.
    """
    np.random.seed(42)
    model = auto_arima(
        data,
        seasonal=False,
        trace=True
    )
    print(model.summary())
    return model


def run_auto_arima_model(
    train: pd.DataFrame, test: pd.DataFrame, target: pd.DataFrame
) -> Tuple[tuple, float, pd.DataFrame]:
    """Run Auto Arima fit and predict functions.

    Parameters:
        train (pd.DataFrame): The training data.
        test (pd.DataFrame): The testing data.
        target (pd.DataFrame): The target column name in the DataFrame.

    Returns:
        model.order (tuple): The order of the ARIMA model.
        model.aic() (float): The Akaike Information Criterion (AIC) value.
        pred_frame (pd.DataFrame): The predicted values as a DataFrame.

    """
    model = autoarima_fit_model(train[target])
    pred = arima_predict(model, test.shape[0])
    pred_frame = pd.DataFrame(pred, columns=["prediction"])
    return model.order, model.aic(), pred_frame


def sarimax_fit_model(data: pd.DataFrame,
                      order: tuple,
                      seasonal_order: tuple,
                      exog_train: pd.DataFrame | None = None) -> SARIMAX:
    """Fit a SARIMAX model to the given order, seasonal order, and data or exog
    For SARIMA in Python, you use statsmodels via SARIMAX.
    Even though it's called SARIMAX, if you don't pass exogenous variables,
    it's just SARIMA.
    Args:
        data (pd.DataFrame): The input data.
        order (tuple): The non-seasonal order (p, d, q).
        seasonal_order (tuple): The seasonal order (P, D, Q, s).
        exog_train (pd.DataFrame | None): Exogenous variables for training.

    Returns:
        SARIMAX: The fitted SARIMA model.
    """
    np.random.seed(42)
    model = SARIMAX(data,
                    exog=exog_train,
                    order=order,
                    seasonal_order=seasonal_order)

    model = model.fit(disp=False,
                      maxiter=10,
                      method='lbfgs',
                      cov_type="none")   # <- faster
    return model


def sarimax_predict(model: SARIMAX, n_periods: int, exog_future: pd.DataFrame | None = None) -> pd.Series:
    """Predict the SARIMAX model.

    Args:
        model (SARIMAX): The fitted SARIMA model.
        n_periods (int): The number of periods to forecast.

    Returns:
        pd.Series: The forecasted values.
    """
    return model.get_forecast(steps=n_periods, exog=exog_future).predicted_mean


def run_sarimax_model(train: pd.DataFrame,
                      test: pd.DataFrame,
                      target: str,
                      order: tuple,
                      seasonal_order: tuple,
                      exog_train: pd.DataFrame | None = None,
                      exog_future: pd.DataFrame | None = None):
    """Run SARIMAX fit and predict functions.

    Args:
        train (pd.DataFrame): The training dataset.
        test (pd.DataFrame): The test dataset.
        target (str): The target column name.
        order (tuple): The non-seasonal order (p, d, q).
        seasonal_order (tuple): The seasonal order (P, D, Q, s).
        exog_train (pd.DataFrame | None): Exogenous variables for training.
        exog_future (pd.DataFrame | None): Exogenous variables for future periods.

    Returns:
        pd.DataFrame: The forecasted values in a DataFrame.
    """
    model = sarimax_fit_model(train[target], order, seasonal_order, exog_train)
    pred = sarimax_predict(model, test.shape[0], exog_future)
    pred_frame = pd.DataFrame({'prediction': pred.values}, index=pred.index)
    return pred_frame


def var_fit_model(data: pd.DataFrame, variables: list[str], p: int) -> VAR:
    """
    Fit a VAR(p) model on the given multivariate training data.

    Args:
        data (pd.DataFrame): Training data.
        variables (list[str]): List of variable names to include in the model.
        p (int): Number of lags to include (order of the model).

    Returns:
        VAR: Fitted VAR model results object.
    """
    model = VAR(data[variables])
    fit = model.fit(p)
    return fit


def var_predict(fit, steps: int) -> np.ndarray:
    """
    Forecast 'steps' ahead using a fitted VAR model.

    Args:
        fit (VAR): Fitted VAR model.
        steps (int): Number of steps to forecast ahead.

    Returns:
        np.ndarray: Array of forecasts.
    """
    p = fit.k_ar
    last_y = fit.endog[-p:]
    return fit.forecast(y=last_y, steps=steps)


def run_var_model(train: pd.DataFrame,
                  test: pd.DataFrame,
                  variables: list[str],
                  p: int,
                  forecast_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Run VAR fit and predict functions.

    Args:
        train (pd.DataFrame): The training dataset.
        test (pd.DataFrame): The test dataset.
        variables (list[str]): List of variable names to include in the model.
        p (int): Number of lags to include (order of the model).
        forecast_index (pd.DatetimeIndex): Index for the forecasted values.

    Returns:
        pd.DataFrame: The forecasted values in a DataFrame.
    """
    fit = var_fit_model(train, variables, p)
    pred = var_predict(fit, len(test))
    pred_frame = pd.DataFrame(pred, columns=variables, index=forecast_index)
    pred_frame.columns = [f"pred_{col}" for col in pred_frame.columns]
    return pred_frame
