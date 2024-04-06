import os
import json
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Dict, Any, Tuple
from utils import log_debug


def get_stock_data(
    ticker: str, start: str, end: str, data_dir: Optional[str] = None
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Fetches stock data for a given ticker within the specified date range. If the data has already been fetched
    and saved, it loads from the saved CSV and JSON files; otherwise, it fetches from YFinance and saves.

    :param ticker: The stock ticker symbol.
    :param start: The start date for the data fetch in 'YYYY-MM-DD' format.
    :param end: The end date for the data fetch in 'YYYY-MM-DD' format.
    :param data_dir: Directory path to save and load the fetched data. If None, it defaults to the current directory.
    :return: A tuple containing the stock data as a Pandas Series and stock info as a dictionary.
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


def normalize_stock_data(data: pd.Series) -> pd.Series:
    """
    Normalizes the stock data to show the percentage change from the first data point.

    :param data: A Pandas Series containing stock data.
    :return: A Pandas Series with the normalized data.
    """
    return (data / data.iloc[0] - 1) * 100


def generate_stock_plotly_chart(
    companies: List[str], companies_investments: List[pd.Series], trendline_type: Optional[str] = None,
    start_date: Optional[str] = None, end_date: Optional[str] = None
) -> go.Figure:
    """
    Generates a line chart for the given companies' investments using Plotly.

    :param companies: A list of company stock tickers.
    :param companies_investments: A list of Pandas Series, each representing the normalized investment values for a company.
    :param trendline_type: Optional; type of trendline to display. Supports 'linear' and 'exponential'.
    :param start_date: The start date for the chart. Used in the chart title.
    :param end_date: The end date for the chart. Used in the chart title.
    :return: A Plotly figure object containing the generated chart.
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


def validate_inputs(
    stock_symbols: List[str], start_date: str, end_date: Optional[str], investment_amount: float,
    investment_percentages: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Validates the input parameters for the stock data retrieval and analysis functions.

    :param stock_symbols: A list of stock ticker symbols.
    :param start_date: Start date for data retrieval in 'YYYY-MM-DD' format.
    :param end_date: End date for data retrieval in 'YYYY-MM-DD' format. If None, defaults to today's date.
    :param investment_amount: Total amount to be invested across the stock symbols.
    :param investment_percentages: Optional; a list of percentages representing how the investment amount is distributed among the stock symbols.
    :return: A dictionary with validated and processed input data.
    """
    # Validate stock symbols
    if not stock_symbols or not all(isinstance(symbol, str) for symbol in stock_symbols):
        raise ValueError("Stock symbols must be a non-empty list of strings.")

    # Validate start date
    try:
        start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        if start_date > datetime.date.today():
            raise ValueError("Start date cannot be in the future.")
    except ValueError:
        raise ValueError("Invalid start date format. Use YYYY-MM-DD.")

    # Validate end date
    if end_date:
        try:
            end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
            if end_date < start_date:
                raise ValueError("End date cannot be before start date.")
            if end_date > datetime.date.today():
                raise ValueError("End date cannot be in the future.")
        except ValueError:
            raise ValueError("Invalid end date format. Use YYYY-MM-DD.")
    else:
        end_date = datetime.date.today()

    # Validate investment amount
    if not isinstance(investment_amount, (int, float)) or investment_amount <= 0:
        raise ValueError("Investment amount must be a positive number.")

    # Validate investment percentages
    if investment_percentages:
        if not all(isinstance(percentage, (int, float)) for percentage in investment_percentages):
            raise ValueError("Investment percentages must be a list of numbers.")
        if len(investment_percentages) != len(stock_symbols):
            raise ValueError("Number of investment percentages must match number of stock symbols.")
        if not (99.9 <= sum(investment_percentages) <= 100.1):  # Allowing a tiny margin for floating point errors
            raise ValueError("Investment percentages must sum up to 100%.")
    else:
        # If no percentages are provided, we assume equal distribution
        investment_percentages = [100 / len(stock_symbols)] * len(stock_symbols)

    return {
        "stock_symbols": stock_symbols,
        "start_date": start_date,
        "end_date": end_date,
        "investment_amount": investment_amount,
        "investment_percentages": investment_percentages
    }


def retrieve_stock_data(stock_symbols: List[str], start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    """
    Retrieves historical stock data for the given symbols from Yahoo Finance API.

    :param stock_symbols: A list of stock ticker symbols.
    :param start_date: The start date for data retrieval.
    :param end_date: The end date for data retrieval.
    :return: A DataFrame with stock prices for the given symbols on start and end dates.
    """
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    # Fetch historical data from Yahoo Finance
    data = yf.download(stock_symbols, start=start_date_str, end=end_date_str)

    # Check if data is empty
    if data.empty:
        raise ValueError("No data fetched. Please check the stock symbols and date range.")

    # Handling for single stock symbol (DataFrame structure is different)
    if len(stock_symbols) == 1:
        data = data['Close'].to_frame()
        data.columns = stock_symbols
    else:
        # We only need the 'Adj Close' prices
        data = data['Adj Close']

    # Function to find the nearest date in the index
    def find_nearest_date(target_date_str, date_index):
        target_date = pd.to_datetime(target_date_str)
        nearest_date = min(date_index, key=lambda x: abs(x - target_date))
        return nearest_date

    # Finding the nearest available dates for start and end dates
    nearest_start_date = find_nearest_date(start_date_str, data.index)
    nearest_end_date = find_nearest_date(end_date_str, data.index)

    # Filter out only the nearest start and end date data
    filtered_data = data.loc[[nearest_start_date, nearest_end_date]]

    return filtered_data


def distribute_investment(
    stock_symbols: List[str], investment_amount: float, investment_percentages: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Distributes the total investment amount among the given stock symbols according to specified percentages.

    :param stock_symbols: A list of stock ticker symbols.
    :param investment_amount: Total amount to be invested.
    :param investment_percentages: A list of percentages specifying how to distribute the investment amount among the stock symbols.
    :return: A dictionary mapping each stock symbol to its allocated investment amount.
    """
    if investment_percentages:
        # Distribute investment based on provided percentages
        investment_distribution = {symbol: amount / 100 * investment_amount
                                   for symbol, amount in zip(stock_symbols, investment_percentages)}
    else:
        # Equal distribution of investment among all stocks
        equal_percentage = 100 / len(stock_symbols)
        investment_distribution = {symbol: equal_percentage / 100 * investment_amount
                                   for symbol in stock_symbols}

    return investment_distribution


def get_portfolio_results(
    stock_symbols: List[str], start_date: str, end_date: Optional[str], investment_amount: float,
    investment_percentages: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Analyzes the investment results for the given stock portfolio over the specified period.

    :param stock_symbols: A list of stock ticker symbols.
    :param start_date: The start date for the investment period.
    :param end_date: The end date for the investment period.
    :param investment_amount: The total amount invested in the portfolio.
    :param investment_percentages: Optional; percentages specifying how the investment amount is distributed among the stocks.
    :return: A dictionary with the analysis results, including individual stock performances and total portfolio value change.
    """

    # Validate input
    validated_inputs = validate_inputs(
        stock_symbols, start_date=start_date, end_date=end_date,
        investment_amount=investment_amount, investment_percentages=investment_percentages
    )

    # get stock data
    stock_data = retrieve_stock_data(
        stock_symbols=validated_inputs['stock_symbols'],
        start_date=validated_inputs['start_date'],
        end_date=validated_inputs['end_date'],
    )

    # Distributed investment
    investment_distribution = distribute_investment(
        validated_inputs['stock_symbols'],
        validated_inputs['investment_amount'],
        validated_inputs['investment_percentages']
    )

    # Ensure end_date is in the correct format for indexing
    end_date_str = validated_inputs['end_date'].strftime("%Y-%m-%d")

    # Calculating portfolio value and individual stock performances
    total_value_end = 0
    results = []
    for symbol, investment in investment_distribution.items():

        # Find the price of the stock on the start and end dates
        start_price = stock_data.loc[stock_data.index[0], symbol]
        end_price = stock_data.loc[stock_data.index[1], symbol]

        # Calculate the number of shares bought and the value of these shares at the end date
        num_shares = investment / start_price
        value_end = num_shares * end_price
        total_value_end += value_end

        # Calculate and display the performance
        performance = ((value_end - investment) / investment) * 100
        results.append({
            'symbol': symbol,
            'investment': investment,
            'end_date': end_date_str,
            'value_end': value_end,
            'performance': performance
        })

    # Return the total value of the portfolio
    return {
        "stocks_data": results,
        "end_portfolio_value": total_value_end,
        "end_portfolio_value_diff": total_value_end - investment_amount,
        "end_portfolio_value_diff_pct": ((total_value_end - investment_amount) / investment_amount) * 100
    }

