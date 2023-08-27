import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_stock_data(ticker, start, end):
    """
    Get stock ticker data using YFinance library for given date range.
    :param ticker: stock ticker
    :param start: start date
    :param end: end date
    :return Pandas series
    """
    stock_data = yf.download(ticker, start=start, end=end)['Adj Close']
    return stock_data


def normalize_stock_data(data):
    """
    Normalize stock data
    :param data: Pandas series
    :return Normalized series
    """
    return (data / data.iloc[0] - 1) * 100


def generate_plotly_chart(company1, company2, investment1, investment2, trendline_type=None):
    """
    Generate line chart using Plotly.
    :param company1: Company 1 stock ticker
    :param company2: Company 2 stock ticker
    :param investment1: Company 1 invested amount series
    :param investment2: Company 2 invested amount series
    :param trendline_type: Chart trend-line type
    :return Plotly figure
    """
    # Initialize Plotly figure
    fig = make_subplots(rows=1, cols=1)

    # Add investment lines for both companies
    fig.add_trace(go.Scatter(x=investment1.index, y=investment1.values, mode='lines', name=f"{company1}"))
    fig.add_trace(go.Scatter(x=investment2.index, y=investment2.values, mode='lines', name=f"{company2}"))

    # Calculate average investment values
    avg_investment1 = np.mean(investment1.values)
    avg_investment2 = np.mean(investment2.values)

    # Add average lines for both companies
    fig.add_trace(go.Scatter(x=investment1.index, y=[avg_investment1] * len(investment1.index),
                             mode='lines', name=f"Avg {company1}", line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=investment2.index, y=[avg_investment2] * len(investment2.index),
                             mode='lines', name=f"Avg {company2}", line=dict(dash='dash')))

    # Add trendlines based on the selected trendline_type
    if trendline_type == "exponential":
        trendline1 = np.polyfit(np.arange(len(investment1)), np.log(investment1.values), 1)
        trendline2 = np.polyfit(np.arange(len(investment2)), np.log(investment2.values), 1)
        fig.add_trace(go.Scatter(x=investment1.index, y=np.exp(np.polyval(trendline1, np.arange(len(investment1)))),
                                 mode='lines', name=f"Exp Trend {company1}", line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=investment2.index, y=np.exp(np.polyval(trendline2, np.arange(len(investment2)))),
                                 mode='lines', name=f"Exp Trend {company2}", line=dict(dash='dot')))
    elif trendline_type == "linear":
        trendline1 = np.polyfit(np.arange(len(investment1)), investment1.values, 1)
        trendline2 = np.polyfit(np.arange(len(investment2)), investment2.values, 1)
        fig.add_trace(go.Scatter(x=investment1.index, y=np.polyval(trendline1, np.arange(len(investment1))),
                                 mode='lines', name=f"Linear Trend {company1}", line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=investment2.index, y=np.polyval(trendline2, np.arange(len(investment2))),
                                 mode='lines', name=f"Linear Trend {company2}", line=dict(dash='dot')))
    # Add more trendline types here (logarithmic, power, polynomial, moving average)

    # Set x-axis ticks and get tick positions
    tickvals = investment1.index[::len(investment1.index) // 10]
    fig.update_xaxes(tickvals=tickvals)

    # Add dollar amount annotations at tick positions
    for tick in tickvals:
        y_val1 = investment1.loc[tick]
        y_val2 = investment2.loc[tick]
        fig.add_annotation(x=tick, y=y_val1, text=f"<b>${y_val1:.0f}</b>", showarrow=False)
        fig.add_annotation(x=tick, y=y_val2, text=f"<b>${y_val2:.0f}</b>", showarrow=False)

    # Label and render the figure
    fig.update_layout(
        yaxis_title='Investment Value (USD)',
        xaxis_title='Date',
        height=700,
        font=dict(
            size=14,
            color="RebeccaPurple"
        )
    )
    return fig
