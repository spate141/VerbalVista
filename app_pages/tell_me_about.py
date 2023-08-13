import time
import streamlit as st
from utils.google_serper_util import GoogleSerper
from streamlit_extras.colored_header import colored_header


def render_tell_me_about_page():
    """

    """
    colored_header(label="Tell me about!", description="Let's ask internet about stuff!", color_name="orange-70")

    gs = GoogleSerper()

    with st.form("tell_me_everything_about"):
        cols = st.columns([0.8, 0.2, 0.3, 0.4, 0.4, 0.3])
        with cols[0]:
            text = st.text_input("What are we looking today?", placeholder="Search query")
        with cols[1]:
            num_results = st.number_input("Total results", max_value=25, min_value=1, value=3, step=1)
        with cols[2]:
            temp = st.number_input("Temperature", max_value=1.0, min_value=0.0, value=0.5)
        with cols[3]:
            model_name = st.selectbox("Model Name", ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"], index=0)
        with cols[4]:
            summ_chain_type = st.selectbox("Chain type", index=1, options=["stuff", "map_reduce", "refine"])
        with cols[5]:
            max_tokens = st.number_input("Max tokens", max_value=1000, min_value=100, value=512)

        submit = st.form_submit_button('Tell me more!', type='primary')

        if submit:
            results = gs.google_serper_summarization(
                search_query=text, num_results=num_results, temperature=temp,
                model_name=model_name, chain_type=summ_chain_type, max_tokens=max_tokens
            )
            for r in results:
                st.markdown(f"**Title: {r['title']}**  \n"
                            f"**Link: {r['link']}**  \n"
                            f"**Summary:**   \n"
                            f"{r['summary']}")
