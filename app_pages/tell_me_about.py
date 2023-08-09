import time
import streamlit as st
from utils.google_serper_util import GoogleSerper
from streamlit_extras.colored_header import colored_header


def render_tell_me_about_page():
    """

    """
    colored_header(
        label="Tell me about!",
        description="Let's ask internet about stuff!",
        color_name="orange-70",
    )
    gs = GoogleSerper()

    with st.form("tell_me_everything_about"):
        cols = st.columns(3)
        with cols[0]:
            text = st.text_input("What are we looking today?")
        with cols[1]:
            temp = st.number_input("Temperature", max_value=1.0, min_value=0.0, value=0.5)
        with cols[2]:
            verbose = st.selectbox("Verbose", [True, False], index=0)
        submit = st.form_submit_button('Tell me more!', type='primary')
        if submit:
            agent = gs.load_google_serper_agent(temperature=temp, verbose=verbose)
            answer = agent.run(text)
            message_placeholder = st.empty()

            # Simulate stream of response with milliseconds delay
            full_response = ""
            for chunk in answer.split():
                full_response += chunk + " "
                time.sleep(0.03)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            # Display full message at the end with other stuff you want to show like `response_meta`.
            message_placeholder.markdown(full_response)

