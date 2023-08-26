import pandas as pd
import streamlit as st
from utils.stocks_util import *


def render_stocks_comparison_page():
    """
    Render stock comparison page.
    """
    st.header('Stock Performance Comparison', divider='red')
    with st.form("stocks"):
        cols = st.columns([1, 1, 1, 1, 1])
        with cols[0]:
            company1 = st.text_input("Enter first company ticker:", "NVDA")
        with cols[1]:
            company2 = st.text_input("Enter second company ticker:", "SPY")
        with cols[2]:
            invested_amount = st.number_input("Invested amount in USD:", min_value=0.0, value=1.0)
        with cols[3]:
            start_date = st.date_input("Start date:", value=pd.to_datetime("2023-01-01"))
        with cols[4]:
            end_date = st.date_input("End date:", value=pd.to_datetime("2023-08-01"))

        submit = st.form_submit_button('Submit')
        if submit:
            # Fetch and normalize stock data
            company1_data_normalized = normalize_stock_data(get_stock_data(company1, start_date, end_date))
            company2_data_normalized = normalize_stock_data(get_stock_data(company2, start_date, end_date))

            # Calculate investment performance
            investment1 = (company1_data_normalized / 100 + 1) * invested_amount
            investment2 = (company2_data_normalized / 100 + 1) * invested_amount

            # Merge data and plot
            merged_data = pd.concat([investment1, investment2], axis=1)
            merged_data.columns = [f"{company1} Investment", f"{company2} Investment"]
            final_investment1, final_investment2 = round(investment1[-1], 2), round(investment2[-1], 2)
            st.markdown(
                f"<b>{company1}:</b> \\${final_investment1} | <b>{company2}:</b> \\${final_investment2}",
                unsafe_allow_html=True
            )
            fig = generate_plotly_chart(company1, company2, investment1, investment2)
            st.plotly_chart(fig, use_container_width=True)
