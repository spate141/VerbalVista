import os
import time
import pickle
import streamlit as st
from utils.logging_module import log_info, log_debug, log_error
from utils.google_serper_util import google_serper_summarization


def render_tell_me_about_page(search_history_dir=None):
    """

    """
    st.header("Tell me about!", divider='grey')
    # load all previous search results directory names
    previous_searches = [None]
    for prev_search_query in os.listdir(search_history_dir):
        item_path = os.path.join(search_history_dir, prev_search_query)
        if os.path.isdir(item_path):
            previous_searches.append(prev_search_query)

    cols = st.columns([0.6, 0.5, 0.2, 0.3, 0.3, 0.3, 0.25])
    with cols[0]:
        search_query = st.text_input("What are we looking today?", placeholder="Search query")
    with cols[1]:
        prev_search_query = st.selectbox("What are we looking today?", options=previous_searches, placeholder="Search query")
    with cols[2]:
        num_results = st.number_input("Total results", max_value=25, min_value=1, value=3, step=1)
    with cols[3]:
        temp = st.number_input("Temperature", max_value=1.0, min_value=0.0, value=0.5)
    with cols[4]:
        model_name = st.selectbox("Model Name", ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"], index=0)
    with cols[5]:
        summ_chain_type = st.selectbox("Chain type", index=1, options=["stuff", "map_reduce", "refine"])
    with cols[6]:
        max_tokens = st.number_input("Max tokens", max_value=1000, min_value=100, value=512)

    submit = st.button('Search', type='primary')
    if submit:

        if search_query == '' and prev_search_query is not None:
            search_query = prev_search_query
        elif prev_search_query is None and search_query != '':
            search_query = search_query
        elif prev_search_query is not None and search_query != '':
            search_query = prev_search_query

        search_result_dir_path = os.path.join(search_history_dir, '_'.join(search_query.split()))
        if not os.path.exists(search_result_dir_path):
            os.makedirs(search_result_dir_path)
            log_debug(f"Directory '{search_result_dir_path}' created successfully.")
        search_history_filepath = os.path.join(
            search_result_dir_path,
            f"{'_'.join(search_query.split())}_num_results-{num_results}.pickle"
        )

        # check if search history is available locally, if yes; load the search history
        if os.path.exists(search_history_filepath):
            log_debug(f"Loading search history from local file: {search_history_filepath}")
            with open(search_history_filepath, 'rb') as f:
                st.session_state[search_history_filepath] = pickle.load(f)

            # display all previous search results
            for r in st.session_state[search_history_filepath]:
                st.markdown(
                    f"**Title: {r['title']}**  \n"
                    f"**Link: {r['link']}**  \n"
                    f"**Summary:**   \n"
                    f"{r['summary']}",
                    unsafe_allow_html=True
                )

        else:
            st.session_state[search_history_filepath] = []
            msg = st.toast('Processing data...')
            results = google_serper_summarization(
                search_query=search_query, num_results=num_results, temperature=temp,
                model_name=model_name, chain_type=summ_chain_type, max_tokens=max_tokens, msg=msg
            )
            for r in results:
                st.markdown(
                    f"**Title: {r['title']}**  \n"
                    f"**Link: {r['link']}**  \n"
                    f"**Summary:**   \n"
                    f"{r['summary']}",
                    unsafe_allow_html=True
                )
                st.session_state[search_history_filepath].append(r)

            # Save conversation to local file
            log_debug(f"Saving search history to local file: {search_history_filepath}")
            with open(search_history_filepath, 'wb') as f:
                pickle.dump(st.session_state[search_history_filepath], f)


