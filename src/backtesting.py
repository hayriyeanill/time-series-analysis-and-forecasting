"""Backtesting module for time series forecasting models.
This module provides functionality to update the training dataset
using backtesting techniques such as sliding window and extending window
approaches."""

from datetime import datetime
import pandas as pd
import numpy as np


def backtesting(
    train_set: pd.DataFrame,
    test_set: pd.DataFrame,
    date: datetime,
    technique: str,
) -> pd.DataFrame:
    """Update the training dataset during backtesting.

    This function simulates a real forecasting scenario by
    adding new observations from the test set into the
    training set step by step.

    Args:
        train_set (pd.DataFrame): Current training data used to fit the model.
        test_set (pd.DataFrame): Test data containing future observations.
        date (datetime): The date whose observations from the test set
                         will be added to the training set.
        technique (str): Backtesting method:
                        - "sliding_window": Adds new data and removes
                        the oldest date from the training set to keep
                        the window size fixed.
                        - "extending_window": Adds new data and keeps
                        all past observations (window grows over time).

    Returns
    pd.DataFrame
        Updated training dataset.
    """
    test_set_first_day_data = test_set[test_set.index.date == date]
    train_set = pd.concat([train_set, test_set_first_day_data])
    if technique == "sliding_window":
        # Remove first date rows in train_set for sliding window effect
        train_set_first_day_data = train_set[
            ((train_set.index).date) == np.unique((train_set.index).date)[0]
        ]
        train_set.drop(train_set_first_day_data.index, inplace=True)
    elif technique == "extending_window":
        pass
    else:
        raise ValueError(
            "Invalid backtesting technique. Use 'sliding_window' or 'extending_window'."
        )
    return train_set
