import os
import time
import pickle
import streamlit as st
from datetime import datetime
from utils import log_info, log_debug, log_error
from utils.rag_utils.rag_util import get_available_indices, load_index_and_metadata, do_some_chat_completion


def render_qa_page(
    temperature=None, max_tokens=None, model_name=None, max_semantic_retrieval_chunks=None,
    max_lexical_retrieval_chunks=None, tx2sp_util=None, indices_dir=None, chat_history_dir=None,
    enable_tts=False, tts_voice=None
):
    """
    This function allow user to do conversation with the data.
    """
    st.header("Q & A", divider='red')
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
        try:
            indexed_data_embedding_model = agent_meta['metadata_dict'][0]['embedding_model'][0]
        except Exception as e:
            log_error(e)
            return
        index_meta = os.path.join(selected_index_path, 'doc.meta.txt')
        index_meta_txt = open(index_meta, 'r').read()
        index_meta_txt = ' '.join(index_meta_txt.split())
        st.markdown(
            f"<b>Model Meta:</b> <font color='#FF7F50'><b>temperature: </b></font><i>{temperature},</i> "
            f"<font color='#DE3163'><b>max_tokens: </b></font><i>{max_tokens},</i> "
            f"<font color='#9A7D0A'><b>llm: </b></font><i>{model_name},</i> "
            f"<font color='#6495ED'><b>embedding: </b></font><i>{indexed_data_embedding_model},</i> "
            f"<font color='#229954'><b>retrieval_chunks: </b></font> <i>semantic: {max_semantic_retrieval_chunks}, lexical: {max_lexical_retrieval_chunks}</i>",
            unsafe_allow_html=True
        )
        st.markdown(f"<h6>Data Description: <i>{index_meta_txt}...</i></h6>", unsafe_allow_html=True)

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
                st.session_state[selected_index_path] = {'messages': [], 'meta': [], 'timestamps': []}

        # Display chat messages from history on app rerun
        for message_item, cost_item, timestamp_item in zip(
            st.session_state[selected_index_path]['messages'], st.session_state[selected_index_path]['meta'],
            st.session_state[selected_index_path]['timestamps']
        ):
            with st.chat_message(message_item["role"], avatar=message_item["role"]):
                st.markdown(message_item["content"])
                if cost_item and timestamp_item:
                    cost_item['utc_time'] = timestamp_item['utc_time']
                    st.json(cost_item, expanded=False)

        # React to user input
        prompt = st.chat_input(f"Start asking questions to '{os.path.basename(selected_index_path)[:50]}...'")

        if prompt:
            # center_css = """
            # <style>
            # div[class*="StatusWidget"]{
            #     position: fixed;
            #     top: 91%;
            #     left: 78%;
            #     transform: translate(-50%, -50%);
            #     width: 50%;
            # }
            # </style>
            # """
            # st.markdown(center_css, unsafe_allow_html=True)

            # Add user message to chat history
            st.session_state[selected_index_path]['messages'].append({
                "role": "user", "content": prompt
            })
            st.session_state[selected_index_path]['meta'].append(None)
            st.session_state[selected_index_path]['timestamps'].append({
                "utc_time": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
            })

            # Display user message in chat message container
            with st.chat_message("user", avatar="human"):
                st.markdown(prompt)

            log_info("QA")
            result = do_some_chat_completion(
                query=prompt, embedding_model=indexed_data_embedding_model, llm_model=model_name,
                temperature=temperature, faiss_index=agent_meta['faiss_index'],
                lexical_index=agent_meta['lexical_index'], metadata_dict=agent_meta['metadata_dict'], reranker=None,
                max_tokens=max_tokens, max_semantic_retrieval_chunks=max_semantic_retrieval_chunks,
                max_lexical_retrieval_chunks=max_lexical_retrieval_chunks
            )
            answer = result['answer']
            answer_meta = result['completion_meta']

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
                st.json(answer_meta, expanded=False)
                if enable_tts:
                    st.audio(tx2sp_util.text_to_speech(text=full_response, voice=tts_voice).content)

            # Add assistant response to chat history
            st.session_state[selected_index_path]['messages'].append({
                "role": "assistant", "content": answer
            })
            st.session_state[selected_index_path]['meta'].append(answer_meta)
            st.session_state[selected_index_path]['timestamps'].append({
                "utc_time": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
            })

            # Save conversation to local file
            log_debug(f"Saving chat history to local file: {chat_history_filepath}")
            with open(chat_history_filepath, 'wb') as f:
                pickle.dump(st.session_state[selected_index_path], f)
            st.rerun()
