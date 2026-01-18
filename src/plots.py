"""Plotting utilities for time series data visualization."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose


def seasonal_decomposition_plot(
    data: pd.DataFrame,
    target: str,
    period: int,
    yaxis_title: str,
    plot_title: str,
):
    """Create a seasonal decomposition plot for a time series.

    Args:
        data (pd.DataFrame): The input DataFrame containing the time series data.
                             The DataFrame index should be datetime-like.
        target (str): Column name of the target variable to decompose
        period (int): The period of the seasonality (e.g., 12 for monthly data with yearly seasonality)
        yaxis_title (str): Title for the y-axis
        plot_title (str): Main title for the plot

    Returns:
        None: Displays the plot directly

    """
    seasonal_decomposition = seasonal_decompose(
        data[target], model="additive", period=period
    )
    trend = seasonal_decomposition.trend
    seasonal = seasonal_decomposition.seasonal
    residual = seasonal_decomposition.resid

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Observed Data", "Trend", "Seasonal", "Residual"),
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data[target],
            mode="lines",
            name="Observed",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=trend,
            mode="lines",
            name="Trend",
            line=dict(color="orange"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=seasonal,
            mode="lines",
            name="Seasonal",
            line=dict(color="green"),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=residual,
            mode="lines",
            name="Residual",
            line=dict(color="red"),
        ),
        row=4,
        col=1,
    )

    fig.update_layout(
        title=plot_title,
        xaxis_title="Date",
        yaxis_title=yaxis_title,
        height=800,
        showlegend=False,
    )
    fig.show()


def plot_line(
    data: pd.DataFrame,
    x_var: str,
    y_var: str,
    xaxis_title: str,
    yaxis_title: str,
    plot_title: str,
):
    """Create a simple line plot.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data to plot
        x_var (str): Column name for x-axis values
        y_var (str): Column name for y-axis values
        xaxis_title (str): Title for the x-axis
        yaxis_title (str): Title for the y-axis
        plot_title (str): Main title for the plot

    Returns:
        None: Displays the plot directly
    """
    fig = px.line(data, x=x_var, y=y_var, title=plot_title)
    fig.update_layout(xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    fig.show()


def plot_line_group(
    data: pd.DataFrame,
    x_var: str,
    y_var: str,
    color: str,
    xaxis_title: str,
    yaxis_title: str,
    plot_title: str,
):
    """Create a grouped line plot with color-coded series.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data to plot
        x_var (str): Column name for x-axis values
        y_var (str): Column name for y-axis values
        color (str): Column name for grouping/coloring the lines
        xaxis_title (str): Title for the x-axis
        yaxis_title (str): Title for the y-axis
        plot_title (str): Main title for the plot

    Returns:
        None: Displays the plot directly
    """
    fig = px.line(data, x=x_var, y=y_var, color=color, title=plot_title)
    fig.update_layout(xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    fig.show()


def plot_bar(
    data: pd.DataFrame,
    x_var: str,
    y_var: str,
    xaxis_title: str,
    yaxis_title: str,
    plot_title: str,
):
    """Create a bar chart.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data to plot
        x_var (str): Column name for x-axis categories
        y_var (str): Column name for y-axis values (bar heights)
        xaxis_title (str): Title for the x-axis
        yaxis_title (str): Title for the y-axis
        plot_title (str): Main title for the plot

    Returns:
        None: Displays the plot directly
    """
    fig = px.bar(data, x=x_var, y=y_var, title=plot_title)
    fig.update_layout(xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    fig.show()


def plot_histogram(
    data: pd.DataFrame,
    x_var: str,
    xaxis_title: str,
    yaxis_title: str,
    plot_title: str,
):
    """Create a histogram to show the distribution of values.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data to plot
        x_var (str): Column name for the variable to create histogram bins
        xaxis_title (str): Title for the x-axis
        yaxis_title (str): Title for the y-axis (typically 'Frequency' or 'Count')
        plot_title (str): Main title for the plot

    Returns:
        None: Displays the plot directly
    """
    fig = px.histogram(data, x=x_var)
    fig.update_layout(
        xaxis_title=xaxis_title, yaxis_title=yaxis_title, title=plot_title
    )
    fig.show()


def seasonal_box_plot(
    data: pd.DataFrame,
    y_var: str,
    xaxis_title: str,
    yaxis_title: str,
    plot_title: str,
):
    """Create a seasonal box plot to visualize distribution across seasons.
    Args:
        data (pd.DataFrame): The input DataFrame containing the data to plot.
                             The DataFrame index should be datetime-like.
        y_var (str): Column name for the variable to plot on the y-axis.
        xaxis_title (str): Title for the x-axis.
        yaxis_title (str): Title for the y-axis.
        plot_title (str): Main title for the plot.
    """
    seasonal_data = data.copy()
    seasonal_data["month"] = seasonal_data.index.month_name()
    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    seasonal_data["month"] = pd.Categorical(
        seasonal_data["month"], categories=month_order, ordered=True
    )
    fig = px.box(seasonal_data, x="month", y=y_var)
    fig.update_layout(
        xaxis_title=xaxis_title, yaxis_title=yaxis_title, title=plot_title
    )
    fig.show()


def plot_heatmap(data: pd.DataFrame):
    """Create a heatmap to visualize data density or intensity.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data to plot
    Returns:
        None: Displays the plot directly
    """

    fig = px.imshow(
        data,
        text_auto=True,
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="Correlation Heatmap of Weather Variables",
    )

    fig.update_layout(
        xaxis_title="Variables",
        yaxis_title="Variables",
        width=1000,
        height=1000,
    )

    fig.show()
