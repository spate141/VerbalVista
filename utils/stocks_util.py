import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_stock_data(ticker, start, end):
    """
    Get stock ticker data using YFinance library for given date range.
    :param ticker: stock ticker
    :param start: start date
    :param end: end date
    """
    stock_data = yf.download(ticker, start=start, end=end)['Adj Close']
    return stock_data


def normalize_stock_data(data):
    """
    Normalize stock data
    """
    return (data / data.iloc[0] - 1) * 100


def generate_plotly_chart(company1, company2, investment1, investment2):
    """

    """
    # Initialize Plotly figure
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=investment1.index, y=investment1.values, mode='lines', name=f"{company1} Investment"))
    fig.add_trace(go.Scatter(x=investment2.index, y=investment2.values, mode='lines', name=f"{company2} Investment"))

    # Set x-axis ticks and get tick positions
    # Here, it takes every nth index where n = total length // 10
    tickvals = investment1.index[::len(investment1.index) // 10]
    fig.update_xaxes(tickvals=tickvals)

    # Add dollar amount annotations at tick positions
    for tick in tickvals:
        y_val1 = investment1.loc[tick]
        y_val2 = investment2.loc[tick]
        fig.add_annotation(x=tick, y=y_val1, text=f"<b>${y_val1:.2f}</b>", showarrow=False)
        fig.add_annotation(x=tick, y=y_val2, text=f"<b>${y_val2:.2f}</b>", showarrow=False)

    # Label and render the figure
    fig.update_layout(
        # title="Investment Performance Comparison:",
        yaxis_title='Investment Value (USD)',
        xaxis_title='Date',
        height=700
    )
    return fig
