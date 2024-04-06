import pandas as pd
import streamlit as st
from typing import Dict
from datetime import date
from utils import log_debug
from utils.other_utils import get_stock_data, normalize_stock_data, generate_stock_plotly_chart


def render_stock_cards(
    company_name: str,
    company_info: Dict[str, float],
    final_investment: float,
    invested_amount: float
) -> None:
    """
    Renders individual stock cards in Streamlit displaying various financial metrics and performance indicators
    for a given company.

    :param company_name: Name of the company.
    :param company_info: Dictionary containing various information about the company, such as current price, target prices, and 52-week high and low.
    :param final_investment: The value of the investment at the end date based on stock performance.
    :param invested_amount: The initial amount invested in the stock.
    """
    c1_cols = st.columns(6)
    with c1_cols[0]:
        st.metric(f"$ {company_name.upper()} $", f"${int(final_investment):,}", f"{int(final_investment - invested_amount):,}")
    with c1_cols[1]:
        st.metric("Current price", f"${company_info.get('currentPrice', company_info.get('open', None)):,}")
    with c1_cols[2]:
        st.metric("Target price HIGH", f"${company_info.get('targetHighPrice', 0):,}")
    with c1_cols[3]:
        st.metric("Target price LOW", f"${company_info.get('targetLowPrice', 0):,}")
    with c1_cols[4]:
        st.metric("52-Week HIGH", f"${company_info['fiftyTwoWeekHigh']:,}")
    with c1_cols[5]:
        st.metric("52-Week LOW", f"${company_info['fiftyTwoWeekLow']:,}")


def render_stocks_comparison_page(stock_data_dir: str = None) -> None:
    """
    Renders a Streamlit page for comparing the performance of different stocks. Users can input the number of
    companies, company tickers, invested amount, start and end dates, and select the trendline type for
    comparison. The page displays individual stock cards for each company and a comparative plot of the investments over time.

    :param stock_data_dir: Directory where stock data is stored. Used to fetch historical stock data.
    """
    st.header('Stock Performance Comparison', divider='red')
    current_date = date.today()
    cols = st.columns([1, 1, 1, 1, 1, 1])
    with cols[0]:
        number_inputs = st.number_input("Numbers of Companies:", value=2, max_value=10, step=1)
    with cols[1]:
        companies = [
            st.text_input(
                label=f'Company Ticker {i+1}:', key=f"text_input_{i}"
            ) for i in range(number_inputs)
        ]
    with cols[2]:
        invested_amount = st.number_input("Invested amount in USD:", min_value=0, value=1000)
    with cols[3]:
        current_year = current_date.year
        first_date_of_year = date(current_year, 1, 1)
        formatted_first_date = first_date_of_year.strftime("%Y-%m-%d")
        start_date = st.date_input("Start date:", value=pd.to_datetime(formatted_first_date))
    with cols[4]:
        formatted_date = current_date.strftime("%Y-%m-%d")
        end_date = st.date_input("End date:", value=pd.to_datetime(formatted_date))
    with cols[5]:
        trendline_type = st.selectbox("Trend-line:", index=0, options=['exponential', 'linear'])
    st.markdown('')
    c = st.columns([1, 0.5, 1])
    with c[1]:
        submit = st.button('💰 Display Data! 💰', type="primary")
    st.markdown('')
    if submit:
        companies_investments = []
        for c in companies:
            log_debug(f"Processing: {c}")
            company_data, company_info = get_stock_data(
                c, start_date, end_date, data_dir=stock_data_dir
            )
            company_data_normalized = normalize_stock_data(company_data)
            investment = (company_data_normalized / 100 + 1) * invested_amount
            companies_investments.append(investment)

            # Render stock cards
            final_investment = round(investment[-1], 2)
            render_stock_cards(c, company_info, final_investment, invested_amount)

        # Merge data and plot
        merged_data = pd.concat(companies_investments, axis=1)
        merged_data.columns = [f"{i} Investment" for i in companies]

        # Plot stock chart
        fig = generate_stock_plotly_chart(
            companies,
            companies_investments,
            trendline_type=trendline_type,
            start_date=start_date,
            end_date=end_date
        )
        st.plotly_chart(fig, use_container_width=True)
