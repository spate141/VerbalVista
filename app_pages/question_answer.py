import os
import time
import pickle
import streamlit as st
from utils import log_info, log_debug
from utils.rag_utils.rag_util import get_available_indices, load_index_and_metadata, do_some_chat_completion


def render_qa_page(
        temperature=None, max_tokens=None, model_name=None, embedding_model_name=None,
        tx2sp_util=None, indices_dir=None, chat_history_dir=None, enable_tts=False, tts_voice=None
):
    """
    This function allow user to do conversation with the data.
    """
    st.header("Q & A", divider='red')
    st.info(f"\n\ntemperature: {temperature}, max_tokens: {max_tokens}, model_name: {model_name}")
    with st.container():
        indices_df = get_available_indices(indices_dir=indices_dir)
        selected_index_path = st.selectbox(
            "Select Index:", options=indices_df['Index Path'].to_list(), index=None,
            placeholder="<select index>", label_visibility="collapsed"
        )
        if selected_index_path is None:
            st.error("Select index first!")
            return

    if selected_index_path is not None:

        # Initialize QA Agent and get chunks for lexical search
        agent_meta = load_index_and_metadata(selected_index_path)
        chunks = [i['text'] for i in agent_meta['metadata_dict'].values()]

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
        # _chat_history = []
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

            # Other Q/A questions
            log_info("QA")
            result = do_some_chat_completion(
                query=prompt, embedding_model=embedding_model_name, llm_model=model_name, temperature=temperature,
                faiss_index=agent_meta['faiss_index'], lexical_index=agent_meta['lexical_index'],
                metadata_dict=agent_meta['metadata_dict'], chunks=chunks, reranker=None
            )
            answer = result['answer']
            question = result['question']
            sources = result['sources']  # List of strings
            answer_meta = None

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
                if enable_tts:
                    st.audio(tx2sp_util.text_to_speech(text=full_response, voice=tts_voice).content)

            # Add assistant response to chat history
            st.session_state[selected_index_path]['messages'].append({
                "role": "assistant", "content": answer
            })
            st.session_state[selected_index_path]['cost'].append(answer_meta)

            # Save conversation to local file
            log_debug(f"Saving chat history to local file: {chat_history_filepath}")
            with open(chat_history_filepath, 'wb') as f:
                pickle.dump(st.session_state[selected_index_path], f)
            # st.rerun()
