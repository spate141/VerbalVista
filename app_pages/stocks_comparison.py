import pandas as pd
import streamlit as st
from datetime import date


def render_stocks_comparison_page(stock_util=None, stock_data_dir=None):
    """
    Render stock comparison page.
    """
    st.header('Stock Performance Comparison', divider='red')
    current_date = date.today()
    with st.form("stocks"):
        cols = st.columns([1, 1, 1, 1, 1, 1])
        with cols[0]:
            company1 = st.text_input("Enter first company ticker:", "NVDA")
        with cols[1]:
            company2 = st.text_input("Enter second company ticker:", "SPY")
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

        submit = st.form_submit_button('Submit', type="primary")
        if submit:
            # Fetch and normalize stock data
            company1_data_normalized = stock_util.normalize_stock_data(
                stock_util.get_stock_data(company1, start_date, end_date, data_dir=stock_data_dir)
            )
            company2_data_normalized = stock_util.normalize_stock_data(
                stock_util.get_stock_data(company2, start_date, end_date, data_dir=stock_data_dir)
            )

            # Calculate investment performance
            investment1 = (company1_data_normalized / 100 + 1) * invested_amount
            investment2 = (company2_data_normalized / 100 + 1) * invested_amount

            # Merge data and plot
            merged_data = pd.concat([investment1, investment2], axis=1)
            merged_data.columns = [f"{company1} Investment", f"{company2} Investment"]
            final_investment1, final_investment2 = round(investment1[-1], 2), round(investment2[-1], 2)
            cols = st.columns([0.8, 0.6, 0.6, 0.8])
            with cols[1]:
                st.metric(company1, f"${int(final_investment1):,}", f"{int(final_investment1-invested_amount):,}")
            with cols[2]:
                st.metric(company2, f"${int(final_investment2):,}", f"{int(final_investment2-invested_amount):,}")
            fig = stock_util.generate_plotly_chart(
                company1, company2,
                investment1, investment2,
                trendline_type=trendline_type,
                start_date=start_date,
                end_date=end_date
            )
            st.plotly_chart(fig, use_container_width=True)
