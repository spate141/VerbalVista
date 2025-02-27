import os
import time
import pickle
import streamlit as st
from datetime import datetime
from typing import Optional, Any, Union
from utils import log_info, log_debug, log_error
from utils.rag_utils.rag_util import get_available_indices, load_index_and_metadata, do_some_chat_completion


def render_qa_page(
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    model_name: Optional[str] = None,
    max_semantic_retrieval_chunks: Union[int, None] = None,
    max_lexical_retrieval_chunks: Union[int, None] = None,
    tx2sp_util: Optional[Any] = None,
    indices_dir: Optional[str] = None,
    chat_history_dir: Optional[str] = None,
    enable_tts: bool = False,
    tts_voice: Optional[str] = None
) -> None:
    """
    Renders a Streamlit page for question and answer interactions using a language model (LLM).
    The page allows users to interact with the model by providing prompts. It also supports text-to-speech
    (TTS) for the responses and keeps a history of the conversation. The function is capable of handling
    interactions with indexed documents for context-enhanced responses.

    :param temperature: Controls randomness in the response generation. Lower values make responses more predictable.
    :param max_tokens: The maximum number of tokens to generate in the response.
    :param model_name: The name of the language model to use for generating responses.
    :param max_semantic_retrieval_chunks: The maximum number of document chunks to retrieve semantically.
    :param max_lexical_retrieval_chunks: The maximum number of document chunks to retrieve lexically.
    :param tx2sp_util: Utility object for text-to-speech conversion.
    :param indices_dir: Directory where indexed documents are stored.
    :param chat_history_dir: Directory to store chat history.
    :param enable_tts: If True, enables text-to-speech for the responses.
    :param tts_voice: The voice ID or model to use for text-to-speech.
    :return: None
    """
    st.header("Ask the LLM!", divider='red')
    with st.container():
        indices_df = get_available_indices(indices_dir=indices_dir)
        data_indices = [None] + indices_df['Index Path'].to_list()
        selected_index_path = st.selectbox(
            "Select Index:", options=data_indices, index=None,
            placeholder="Ask the LLM or select index to provide additional context!", label_visibility="collapsed"
        )

    if selected_index_path is not None:
        # Chat with RAG index
        rerun = True
        total_cost = 0
        # Initialize QA Agent and get chunks for lexical search
        agent_meta = load_index_and_metadata(selected_index_path)
        try:
            indexed_data_embedding_model = agent_meta['metadata_dict'][0]['embedding_model']
        except Exception as e:
            log_error(e)
            return
        index_meta = os.path.join(selected_index_path, 'doc.meta.txt')
        index_meta_txt = open(index_meta, 'r').read()
        index_meta_txt = ' '.join(index_meta_txt.split())
        st.markdown(
            f"<b>LLM Setting:</b> <font color='#FF7F50'><b>temperature: </b></font><i>{temperature},</i> "
            f"<font color='#DE3163'><b>max_tokens: </b></font><i>{max_tokens},</i> "
            f"<font color='#9A7D0A'><b>llm: </b></font><i>{model_name},</i> "
            f"<font color='#6495ED'><b>embedding: </b></font><i>{indexed_data_embedding_model},</i> "
            f"<font color='#229954'><b>retrieval_chunks: </b></font> <i>semantic: {max_semantic_retrieval_chunks}, lexical: {max_lexical_retrieval_chunks}</i>",
            unsafe_allow_html=True
        )
        if index_meta_txt:
            cols = st.columns([1, 2, 1])
            with cols[1]:
                st.markdown(f"<h5> {index_meta_txt}... </h5>", unsafe_allow_html=True)

        chat_dir_path = os.path.join(chat_history_dir, os.path.basename(selected_index_path))
        if not os.path.exists(chat_dir_path):
            os.makedirs(chat_dir_path)
            log_debug(f"Directory '{chat_dir_path}' created successfully.")
        chat_history_filepath = os.path.join(chat_dir_path, f"{os.path.basename(selected_index_path)}.pickle")

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
            st.session_state[selected_index_path]['messages'],
            st.session_state[selected_index_path]['meta'],
            st.session_state[selected_index_path]['timestamps']
        ):
            with st.chat_message(message_item["role"], avatar=message_item["role"]):
                st.markdown(message_item["content"])
                if cost_item and timestamp_item:
                    cost_item['utc_time'] = timestamp_item['utc_time']
                    st.caption(f'TOKENS: {{total: {cost_item["tokens"]["total"]}, prompt: {cost_item["tokens"]["prompt"]}, '
                               f'completion: {cost_item["tokens"]["completion"]}}} | '
                               f'COST: {{total: \${round(cost_item["cost"]["total"], 4)}, '
                               f'prompt: \${round(cost_item["cost"]["prompt"], 4)}, '
                               f'completion: \${round(cost_item["cost"]["completion"], 4)}}}')
                    st.json(cost_item, expanded=False)
                    total_cost += cost_item['cost']['total']

        # React to user input
        prompt = st.chat_input(f"Start asking questions to '{os.path.basename(selected_index_path)[:50]}...'")

        if prompt:

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
            total_cost += answer_meta['cost']['total']

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
                    rerun = False
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
            if rerun:
                st.rerun()

        st.markdown(f'<b>Total Q&A Cost: <i>${round(total_cost, 4)}</i></b>', unsafe_allow_html=True)

    else:
        # Chat with LLM directly!
        st.markdown(
            f"<b>LLM Setting:</b> <font color='#FF7F50'><b>temperature: </b></font><i>{temperature},</i> "
            f"<font color='#DE3163'><b>max_tokens: </b></font><i>{max_tokens},</i> "
            f"<font color='#9A7D0A'><b>llm: </b></font><i>{model_name}</i>",
            unsafe_allow_html=True
        )
        rerun = True
        total_cost = 0
        # embedding_model = 'text-embedding-3-small'
        now = datetime.now()
        formatted_date_time = now.strftime('%m%d%Y')
        selected_index_path = f'chat-{formatted_date_time}'
        chat_dir_path = os.path.join(chat_history_dir, selected_index_path)
        if not os.path.exists(chat_dir_path):
            os.makedirs(chat_dir_path)
            log_debug(f"Directory '{chat_dir_path}' created successfully.")
        chat_history_filepath = os.path.join(chat_dir_path, f'{selected_index_path}.pickle')

        if selected_index_path not in st.session_state:
            if os.path.exists(chat_history_filepath):
                log_debug(f"Loading chat history from local file: {chat_history_filepath}")
                with open(chat_history_filepath, 'rb') as f:
                    st.session_state[selected_index_path] = pickle.load(f)
            else:
                st.session_state[f'chat-{formatted_date_time}'] = {'messages': [], 'meta': [], 'timestamps': []}

        # Display chat messages from history on app rerun
        for message_item, cost_item, timestamp_item in zip(
            st.session_state[selected_index_path]['messages'],
            st.session_state[selected_index_path]['meta'],
            st.session_state[selected_index_path]['timestamps']
        ):
            with st.chat_message(message_item["role"], avatar=message_item["role"]):
                st.markdown(message_item["content"])
                if cost_item and timestamp_item:
                    cost_item['utc_time'] = timestamp_item['utc_time']
                    st.caption(
                        f'TOKENS: {{total: {cost_item["tokens"]["total"]}, prompt: {cost_item["tokens"]["prompt"]}, '
                        f'completion: {cost_item["tokens"]["completion"]}}} | '
                        f'COST: {{total: \${round(cost_item["cost"]["total"], 4)}, '
                        f'prompt: \${round(cost_item["cost"]["prompt"], 4)}, '
                        f'completion: \${round(cost_item["cost"]["completion"], 4)}}}')
                    st.json(cost_item, expanded=False)
                    total_cost += cost_item['cost']['total']

        # React to user input
        prompt = st.chat_input(f"Ask your question!")

        if prompt:
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

            log_info("Normal Chat")
            result = do_some_chat_completion(
                query=prompt, llm_model=model_name, temperature=temperature, max_tokens=max_tokens,
                embedding_model='', faiss_index=None, lexical_index=None, metadata_dict=None, reranker=None,
                max_semantic_retrieval_chunks=0, max_lexical_retrieval_chunks=0
            )
            answer = result['answer']
            answer_meta = result['completion_meta']
            total_cost += answer_meta['cost']['total']

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
                    rerun = False
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
            if rerun:
                st.rerun()

        st.markdown(f'<b>Total Q&A Cost: <i>${round(total_cost, 4)}</i></b>', unsafe_allow_html=True)
