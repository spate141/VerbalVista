import pandas as pd
import streamlit as st
from datetime import date
from utils import log_debug
from utils.other_utils import get_portfolio_results


def render_stock_cards(company_name, value_end, invested_amount):
    """
    :param company_name: Company Name
    :param value_end: Final invested amount on the end date.
    :param invested_amount: Initial invested amount on start date.
    """
    c1_cols = st.columns(1)
    with c1_cols[0]:
        st.metric(
            f"{company_name.upper()}", f"${int(value_end):,}", f"{int(value_end - invested_amount):,}"
        )


def render_stocks_portfolio_page():
    """
    Render stock portfolio page.
    """
    st.header('Stock Performance Comparison', divider='red')
    current_date = date.today()
    cols = st.columns([1, 1, 1, 1])
    with cols[0]:
        companies = st.text_area(
            label=f'Company Tickers (separated by ,):', key=f"tickers_input"
        )
        companies = [i.strip().upper() for i in companies.split(',')]
        log_debug(f'Companies: {companies}')
    with cols[1]:
        invested_amount = st.number_input("Invested amount in USD:", min_value=0, value=1000)
    with cols[2]:
        current_year = current_date.year
        first_date_of_year = date(current_year, 1, 1)
        formatted_first_date = first_date_of_year.strftime("%Y-%m-%d")
        start_date = st.date_input("Start date:", value=pd.to_datetime(formatted_first_date))
        start_date = start_date.strftime("%Y-%m-%d")
    with cols[3]:
        formatted_date = current_date.strftime("%Y-%m-%d")
        end_date = st.date_input("End date:", value=pd.to_datetime(formatted_date))
        end_date = end_date.strftime("%Y-%m-%d")
    st.markdown('')
    c = st.columns([1, 0.5, 1])
    with c[1]:
        submit = st.button('💰 Display Data! 💰', type="primary")
    st.markdown('')
    if submit:
        results = get_portfolio_results(
            companies, start_date=start_date, end_date=end_date,
            investment_amount=invested_amount, investment_percentages=None
        )
        st.json(results, expanded=False)
        render_stock_cards('Final Portfolio', results['end_portfolio_value'], invested_amount)
        for r in results['stocks_data']:
            symbol = r['symbol']
            investment = r['investment']
            end_date = r['end_date']
            value_end = r['value_end']
            performance = r['performance']
            render_stock_cards(symbol, value_end, investment)

