import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import log_debug


def get_stock_data(ticker, start, end, data_dir=None):
    """
    Get stock ticker data using YFinance library for given date range.
    :param ticker: stock ticker
    :param start: start date
    :param end: end date
    :param data_dir: Stock save data dir
    :return Pandas series
    """
    csv_file_path = os.path.join(data_dir, f"{ticker}_{start}_{end}.csv")
    json_file_path = os.path.join(data_dir, f"{ticker}_{start}_{end}.json")
    if os.path.exists(csv_file_path) and os.path.exists(json_file_path):
        log_debug(f"Loading {ticker} data from file: {csv_file_path}")
        stock_data = pd.read_csv(csv_file_path, parse_dates=True)
        stock_data = pd.Series(stock_data['Adj Close'].values, index=pd.to_datetime(stock_data['Date']))
        with open(json_file_path, "r") as f:
            stock_info = json.load(f)
    else:
        log_debug(f"Loading {ticker} from YFinance!")
        stock_data = yf.download(ticker, start=start, end=end)['Adj Close']
        stock_info = yf.Ticker(ticker).info
        stock_data.to_csv(csv_file_path)
        with open(json_file_path, 'w') as f:
            json.dump(stock_info, f, indent=2)

    return stock_data, stock_info


def normalize_stock_data(data):
    """
    Normalize stock data
    :param data: Pandas series
    :return Normalized series
    """
    return (data / data.iloc[0] - 1) * 100


def generate_stock_plotly_chart(companies, companies_investments, trendline_type=None, start_date=None, end_date=None):
    """
    Generate line chart using Plotly.
    :param companies: List of company stock tickers
    :param companies_investments: List of companies' invested amounts as series
    :param trendline_type: Chart trend-line type
    :param start_date: Start date
    :param end_date: End date
    :return Plotly figure
    """
    fig = make_subplots(rows=1, cols=1)
    for company, investments in zip(companies, companies_investments):
        fig.add_trace(go.Scatter(x=investments.index, y=investments.values, mode='lines', name=f"{company}"))
        avg_investment = np.mean(investments.values)
        fig.add_trace(go.Scatter(
            x=investments.index, y=[avg_investment] * len(investments.index),
            mode='lines', name=f"Avg {company}", line=dict(dash='dash'))
        )

        # Add trendline based on the selected trendline_type
        if trendline_type:
            if trendline_type == "exponential":
                trendline = np.polyfit(np.arange(len(investments)), np.log(investments.values), 1)
                trendline_label = "Exp Trend"
                trendline_dash = "dot"
            elif trendline_type == "linear":
                trendline = np.polyfit(np.arange(len(investments)), investments.values, 1)
                trendline_label = "Linear Trend"
                trendline_dash = "dot"
            # Add more trendline types here (logarithmic, power, polynomial, moving average)

            trendline_values = np.polyval(trendline, np.arange(len(investments)))
            fig.add_trace(go.Scatter(
                x=investments.index, y=trendline_values,
                mode='lines', name=f"{trendline_label} {company}",
                line=dict(dash=trendline_dash)
            ))

    # Set x-axis ticks and get tick positions
    tickvals = investments.index[::len(investments.index) // 10]
    fig.update_xaxes(tickvals=tickvals)

    # Add dollar amount annotations at tick positions
    for tick in tickvals:
        for company, investments in zip(companies, companies_investments):
            y_val = investments.loc[tick]
            fig.add_annotation(x=tick, y=y_val, text=f"<b>${y_val:.0f}</b>", showarrow=False)

    # Label and render the figure
    fig.update_layout(
        title_text=f"{' Vs '.join(companies)}: {start_date} to {end_date}",
        title_x=0.35,
        title_y=0.9,
        title_font_size=24,
        title_font_color="Black",
        yaxis_title='Investment Value (USD)',
        xaxis_title='Date',
        height=700,
        font_size=14,
        font_color="RebeccaPurple",
    )
    return fig
