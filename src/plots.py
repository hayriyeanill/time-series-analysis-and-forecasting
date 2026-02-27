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


def plot_bar_group(data: pd.DataFrame,
                   x_var: str, y_var: str,
                   color: str, text: str,
                   plot_title: str):
    """Plot bar chart for compare groups.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data to plot
        x_var (str): Column name for x-axis categories
        y_var (str): Column name for y-axis values (bar heights)
        color (str): Column name for grouping/coloring the bars
        text (str): Column name for bar labels
        plot_title (str): Main title for the plot

    Returns:
        None: Displays the plot directly
    """
    fig = px.bar(data,
                 x=x_var,
                 y=y_var,
                 color=color,
                 text=text,
                 title=plot_title,
                 barmode='group')
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


def compare_performance_by_horizon(sliding_performance: pd.DataFrame,
                                   extending_performance: pd.DataFrame,
                                   metrics: list,
                                   model_name: str) -> None:
    """Plot comparison of performance metrics by forecast horizon
    for sliding and extending window backtesting techniques for
    one model.

    Args:
        sliding_performance (pd.DataFrame): DataFrame containing performance metrics for sliding window technique.
        extending_performance (pd.DataFrame): DataFrame containing performance metrics for extending window technique.
        metrics (list): List of metric names to compare.

    Returns:
        None: Displays the plots comparing performance metrics by horizon.
    """
    for metric in metrics:
        fig = px.line()
        fig.add_scatter(
            x=sliding_performance["horizon"],
            y=sliding_performance[metric],
            mode='lines+markers',
            name='Sliding Window',
            line=dict(color='blue')
        )

        fig.add_scatter(
            x=extending_performance["horizon"],
            y=extending_performance[metric],
            mode='lines+markers',
            name='Extending Window',
            line=dict(color='red')
        )
        if metric == "Bias (°C)":
            fig.add_trace(
                go.Scatter(x=sliding_performance["horizon"],
                           y=[0] * len(sliding_performance["horizon"]),
                           mode='lines',
                           line=dict(color='black', width=2),
                           name='Zero Bias Line'))

        fig.update_layout(
            title=f'{metric} Performance by Horizon for {model_name} Model',
            xaxis_title='Horizon',
            yaxis_title=metric,
            legend_title='Backtesting Technique'
        )
        fig.show()


def compare_models_by_horizon(model_performances: dict,
                              metrics: list,
                              technique: str) -> None:
    """Plot comparison of performance metrics by forecast horizon
    for multiple models.

    Args:
        model_performances (dict): Dictionary with model names as keys and performance DataFrames as values. 
        metrics (list): List of metric names to compare.
        technique (str): Backtesting technique name (e.g., 'Sliding Window', 'Extending Window').

    Returns:
        None: Displays the plots comparing performance metrics by horizon.
    """
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

    for metric in metrics:
        fig = px.line()

        for idx, (model_name, perf_df) in enumerate(model_performances.items()):
            fig.add_scatter(
                x=perf_df["horizon"],
                y=perf_df[metric],
                mode='lines+markers',
                name=model_name,
                line=dict(color=colors[idx % len(colors)])
            )

        if metric == "Bias (°C)":
            first_perf = list(model_performances.values())[0]
            fig.add_trace(
                go.Scatter(x=first_perf["horizon"],
                           y=[0] * len(first_perf["horizon"]),
                           mode='lines',
                           line=dict(color='black', width=2),
                           name='Zero Bias Line'))

        fig.update_layout(
            title=f'{metric} Performance by Horizon ({technique})',
            xaxis_title='Horizon',
            yaxis_title=metric,
            legend_title='Model'
        )
        fig.show()


def compare_techniques_by_model(model_performances: dict,
                                metric: str) -> None:
    """Compare sliding vs extending window techniques across models
    for each horizon.

    Args:
        model_performances (dict): Nested dict with structure:
        metric (str): Metric to compare (default: 'MAE (°C)')

    Returns:
        None: Displays the plot
    """
    # Get all horizons from first model
    first_model = list(model_performances.values())[0]
    first_technique = list(first_model.values())[0]
    horizons = first_technique['horizon'].unique()

    for horizon in horizons:
        data = []
        for model_name, techniques in model_performances.items():
            for technique_name, perf_df in techniques.items():
                horizon_data = perf_df[perf_df['horizon'] == horizon]
                data.append({
                    'Model': model_name,
                    'Technique': technique_name,
                    metric: horizon_data[metric].values[0]
                })

        df = pd.DataFrame(data)
        fig = px.bar(df, x='Model', y=metric, color='Technique',
                     barmode='group',
                     title=f'{metric} Comparison at {horizon}')
        fig.show()
