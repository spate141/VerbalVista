import os
import time
import pickle
import streamlit as st
from utils.logging_module import log_info, log_debug, log_error


def render_qa_page(
        temperature=None, max_tokens=None, model_name=None, chain_type=None,
        ask_util=None, indexing_util=None, summary_util=None,
        indices_dir=None, document_dir=None, chat_history_dir=None
):
    """
    This function allow user to do conversation with the data.
    """

    def _gen_summary(t):
        keywords = ["summarize", "summary"]
        for keyword in keywords:
            if keyword in t.lower():
                return True
        return False

    # icons = {"user": "docs/user.png", "assistant": "docs/robot.png"}
    st.header("Q & A", divider='red')
    st.info(f"\n\ntemperature: {temperature}, max_tokens: {max_tokens}, model_name: {model_name}")
    with st.container():
        # enable_audio = st.checkbox("Enable TTS")
        indices_df = indexing_util.get_available_indices(indices_dir=indices_dir)
        selected_index_path = st.selectbox(
            "Select Index:", options=indices_df['Index Path'].to_list(), index=None,
            placeholder="<select index>", label_visibility="collapsed"
        )

        if selected_index_path is None:
            st.error("Select index first!")
            return

    if selected_index_path is not None:

        # Initialize Q/A Chain
        qa_chain = ask_util.prepare_qa_chain(
            index_directory=selected_index_path,
            temperature=temperature,
            model_name=model_name,
            max_tokens=max_tokens
        )
        # Initialize summarization Chain
        summarization_chain = summary_util.initialize_summarization_chain(
            temperature=temperature, max_tokens=max_tokens, chain_type=chain_type
        )

        index_meta = os.path.join(selected_index_path, 'doc.meta.txt')
        index_meta_txt = open(index_meta, 'r').read()
        index_meta_txt = ' '.join(index_meta_txt.split())
        st.success(f"Description: {index_meta_txt}")
        chat_dir_path = os.path.join(chat_history_dir, os.path.basename(selected_index_path))
        if not os.path.exists(chat_dir_path):
            os.makedirs(chat_dir_path)
            log_debug(f"Directory '{chat_dir_path}' created successfully.")
        chat_history_filepath = os.path.join(chat_dir_path, f"{os.path.basename(selected_index_path)}.pickle")

        # Initialize chat history
        _chat_history = []
        if selected_index_path not in st.session_state:

            # check if chat history is available locally, if yes; load the chat history
            if os.path.exists(chat_history_filepath):
                log_debug(f"Loading chat history from local file: {chat_history_filepath}")
                with open(chat_history_filepath, 'rb') as f:
                    st.session_state[selected_index_path] = pickle.load(f)
            else:
                st.session_state[selected_index_path] = {'messages': [], 'cost': []}

        # Display chat messages from history on app rerun
        for message_item, cost_item in zip(
                st.session_state[selected_index_path]['messages'], st.session_state[selected_index_path]['cost']
        ):
            with st.chat_message(message_item["role"], avatar=message_item["role"]):
                st.markdown(message_item["content"])
                if cost_item:
                    st.info(cost_item)
                    # st.info(total_qa_cost)
        cols = st.columns(8)
        with cols[0]:
            get_one_point_summary = st.button("Highlight", type="primary")
        with cols[1]:
            get_summary = st.button("Summary", type="primary")

        if get_summary:
            # React to summarize button
            prompt = st.chat_input(f"Start asking questions to '{os.path.basename(selected_index_path)}'")
            prompt = "Generate summary"
        elif get_one_point_summary:
            prompt = st.chat_input(f"Start asking questions to '{os.path.basename(selected_index_path)}'")
            prompt = "Describe this document in a single bullet point."
        else:
            # React to user input
            prompt = st.chat_input(f"Start asking questions to '{os.path.basename(selected_index_path)}'")

        if prompt:
            center_css = """
            <style>
            div[class*="StatusWidget"]{
                position: fixed;
                top: 50%;
                left: 60%;
                transform: translate(-50%, -50%);
                width: 50%;
            }
            </style>
            """
            st.markdown(center_css, unsafe_allow_html=True)
            # Add user message to chat history
            st.session_state[selected_index_path]['messages'].append({
                "role": "user", "content": prompt
            })
            st.session_state[selected_index_path]['cost'].append(None)

            # Display user message in chat message container
            with st.chat_message("user", avatar="human"):
                st.markdown(prompt)

            if _gen_summary(prompt) or get_summary:
                # If the prompt is asking to summarize
                log_info("Summarization")
                doc_filepath = os.path.join(
                    document_dir, os.path.basename(selected_index_path),
                    f"{os.path.basename(selected_index_path)}.data.txt"
                )
                with open(doc_filepath, 'r') as f:
                    text = f.read()
                answer, answer_meta, chat_history = summary_util.summarize(
                    chain=summarization_chain, text=text, question=prompt,
                    chat_history=_chat_history
                )
                # Display assistant response in chat message container
                with st.chat_message("ai", avatar="ai"):
                    st.markdown(answer)
                    st.info(answer_meta)
                    # st.info(total_qa_cost)
            else:
                # Other Q/A questions
                log_info("QA")
                answer, answer_meta, chat_history = ask_util.ask_question(
                    question=prompt, qa_chain=qa_chain, chat_history=_chat_history
                )
                # Display assistant response in chat message container
                with st.chat_message("ai", avatar="ai"):
                    message_placeholder = st.empty()
                    full_response = ""

                    # Simulate stream of response with milliseconds delay
                    for chunk in answer.split():
                        full_response += chunk + " "
                        time.sleep(0.03)

                        # Add a blinking cursor to simulate typing
                        message_placeholder.markdown(full_response + "▌")

                    # Display full message at the end with other stuff you want to show like `response_meta`.
                    message_placeholder.markdown(full_response)
                    st.info(answer_meta)

            _chat_history.extend(chat_history)

            # Add assistant response to chat history
            st.session_state[selected_index_path]['messages'].append({
                "role": "assistant", "content": answer
            })
            st.session_state[selected_index_path]['cost'].append(answer_meta)

            # Save conversation to local file
            log_debug(f"Saving chat history to local file: {chat_history_filepath}")
            with open(chat_history_filepath, 'wb') as f:
                pickle.dump(st.session_state[selected_index_path], f)
