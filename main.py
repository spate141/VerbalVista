import os
import time
import spacy
import pickle
import pandas as pd
import numpy as np
import streamlit as st
# from TTS.api import TTS
from spacy_streamlit import visualize_ner
from streamlit_extras.colored_header import colored_header
from streamlit_extras.no_default_selectbox import selectbox
from utils.ask_util import AskUtil
from utils.indexing_util import IndexUtil
from utils.summary_util import SummaryUtil
from utils.generate_wordcloud import generate_wordcloud
from utils.audio_transcribe import WhisperAudioTranscribe
from utils.logging_module import log_info, log_debug, log_error
from utils.document_parser import parse_docx, parse_pdf, parse_txt, parse_email, parse_url, write_data_to_file


class VerbalVista:

    def __init__(self, document_dir: str = None, tmp_audio_dir: str = None, indices_dir: str = None, chat_history_dir: str = None):
        self.whisper = WhisperAudioTranscribe()
        self.indexing_util = IndexUtil()
        self.ask_util = AskUtil()
        self.summary_util = SummaryUtil()

        # model_name = 'tts_models/en/ljspeech/tacotron2-DDC'
        # self.tts = TTS(model_name=model_name, progress_bar=True, gpu=False)

        _ = [self.create_directory(d) for d in [document_dir, indices_dir, tmp_audio_dir, chat_history_dir]]
        self.document_dir = document_dir
        self.indices_dir = indices_dir
        self.tmp_audio_dir = tmp_audio_dir
        self.chat_history_dir = chat_history_dir
        self.nlp = spacy.load("en_core_web_sm")
        self.ner_labels = self.nlp.get_pipe("ner").labels

    @staticmethod
    def create_directory(directory_path):
        """

        :param directory_path:
        :return:
        """
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            log_debug(f"Directory '{directory_path}' created successfully.")

    @staticmethod
    def remove_temp_files(directory):
        """

        :param directory:
        :return:
        """
        for file_name in os.listdir(directory):
            file_path = os.path.join(directory, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

    def render_media_processing_page(self):
        """
        """
        supported_formats = ['m4a', 'mp3', 'wav', 'webm', 'mp4', 'mpg', 'mpeg', 'docx', 'pdf', 'txt', 'eml']
        colored_header(
            label="Process Media",
            description=f"Process audio, text, documents and emails.",
            color_name="violet-70",
        )
        with st.form('docs_processing'):

            st.markdown(f"<h6>Process file:</h6>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Upload file:", type=supported_formats, accept_multiple_files=False,
                label_visibility="collapsed"
            )

            st.markdown("<h6>Extract text from URL:</h6>", unsafe_allow_html=True)
            url = st.text_input("Enter URL:", placeholder='https://YOUR_URL', label_visibility="collapsed")
            url = None if len(url) == 0 else url

            st.markdown("<h6>Copy/Paste text:</h6>", unsafe_allow_html=True)
            text = st.text_area("Paste text:", placeholder='YOUR TEXT', label_visibility="collapsed")
            text = None if len(text) == 0 else text

            submitted = st.form_submit_button("Process", type="primary")
            if submitted:
                full_document = ''

                if uploaded_file is not None:
                    log_debug('Processing uploaded file!')
                    if uploaded_file.name.endswith(('.m4a', '.mp3', '.wav', '.webm', '.mp4', '.mpga', '.mpeg')):
                        with st.spinner('Processing audio. Please wait.'):
                            process_audio_bar = st.progress(0, text="Processing...")
                            # Save the uploaded file to the specified directory
                            tmp_audio_save_path = os.path.join(self.tmp_audio_dir, uploaded_file.name)
                            log_debug(f"tmp_save_path: {tmp_audio_save_path}")
                            with open(tmp_audio_save_path, "wb") as f:
                                f.write(uploaded_file.getvalue())
                            # Generate audio chunks
                            audio_chunks_files, file_size_mb, file_duration_in_ms = self.whisper.generate_audio_chunks(
                                audio_filepath=tmp_audio_save_path, max_audio_size=25, tmp_dir=self.tmp_audio_dir,
                                process_bar=process_audio_bar
                            )
                            st.markdown(f"""
                            #### Audio Meta:
                            - Audio file size: {round(file_size_mb, 2)} MB
                            - Audio file duration: {self.whisper.convert_milliseconds(int(file_duration_in_ms))}
                            """)

                        # Get transcript
                        start = time.time()
                        full_document = []
                        with st.spinner('Transcribing audio. Please wait.'):
                            transcribe_audio_bar = st.progress(0, text="Transcribing...")
                            total_chunks = len(audio_chunks_files)
                            pct_cmp = [i / total_chunks for i in range(1, total_chunks + 1)]
                            for index, i in enumerate(audio_chunks_files):
                                transcript = self.whisper.transcribe_audio(i)
                                full_document.append(transcript)
                                transcribe_audio_bar.progress(
                                    pct_cmp[index - 1], f'Audio transcribed: {round(time.time()-start,2)} sec'
                                )
                            full_document = ' '.join(full_document)

                        log_debug(f"Removing tmp audio files")
                        self.remove_temp_files(self.tmp_audio_dir)

                    elif uploaded_file.name.endswith(".pdf"):
                        with st.spinner('Processing pdf file. Please wait.'):
                            full_document = parse_pdf(uploaded_file)

                    elif uploaded_file.name.endswith(".docx"):
                        with st.spinner('Processing word file. Please wait.'):
                            full_document = parse_docx(uploaded_file)

                    elif uploaded_file.name.endswith(".txt"):
                        with st.spinner('Processing text file. Please wait.'):
                            full_document = parse_txt(uploaded_file)

                    elif uploaded_file.name.endswith(".eml"):
                        # Save the uploaded file to the specified directory
                        with st.spinner('Processing email file. Please wait.'):
                            full_document = parse_email(uploaded_file)

                    uploaded_file_name = uploaded_file.name.replace('.', '_').replace(' ', '_')

                elif url is not None:
                    log_debug('Processing URL!')
                    full_document = parse_url(url)
                    uploaded_file_name = url[8:].replace("/", "-").replace('.', '-')

                elif text is not None:
                    log_debug('Processing Text!')
                    full_document = text
                    uploaded_file_name = text[:20].replace("/", "-").replace('.', '-')

                else:
                    st.error("You have to either upload a file, URL or enter some text!")
                    return

                if len(full_document) == 0:
                    st.error("No content available! Try something else.")
                    return

                else:
                    # Write document to a file
                    st.markdown("#### Document snippet:")
                    st.caption(full_document[:110] + '...')
                    tmp_document_save_path = write_data_to_file(
                        uploaded_file_name=uploaded_file_name,
                        document_dir=self.document_dir,
                        full_document=full_document
                    )
                    st.success(f"Document saved: {tmp_document_save_path}")

    def render_manage_index_page(self):
        """
        """
        colored_header(
            label="Manage Index",
            description="Manage documents indices.",
            color_name="blue-green-70",
        )

        st.markdown("<h6>Select Mode:</h6>", unsafe_allow_html=True)
        mode = st.selectbox("mode", ["Create", "Delete"], index=0, label_visibility="collapsed")
        mode_label = None

        if mode == "Create":
            mode_label = 'Creating'
            # st.markdown(
            #     "<h6>LangChain PromptHelper Parameters:</h6>", unsafe_allow_html=True
            # )
            cols = st.columns(2)
            with cols[0]:
                st.markdown("<h6>Select Embedding Model:</h6>", unsafe_allow_html=True)
                embedding_model = st.selectbox("embedding_model:", options=[
                    "text-embedding-ada-002"
                ], index=0, label_visibility="collapsed")
            with cols[1]:
                st.markdown("<h6>Chunk Size:</h6>", unsafe_allow_html=True)
                chunk_size = st.number_input("chunk_size:", value=600, label_visibility="collapsed")
            st.markdown("</br>", unsafe_allow_html=True)

        elif mode == "Delete":
            mode_label = 'Deleting'
            pass

        st.markdown("<h6>Available Documents:</h6>", unsafe_allow_html=True)
        documents_df = self.indexing_util.get_available_documents(
            document_dir=self.document_dir, indices_dir=self.indices_dir
        )
        documents_df['Creation Date'] = pd.to_datetime(documents_df['Creation Date'])
        documents_df = documents_df.sort_values(by='Creation Date', ascending=False)
        selected_documents_df = st.data_editor(documents_df, hide_index=True, use_container_width=True)

        submit = st.button("Submit", type="primary")
        if submit:
            _, c, _ = st.columns([2, 5, 2])
            with c:
                with st.spinner(f'{mode_label} document. Please wait.'):
                    document_dirs = selected_documents_df[selected_documents_df['Select Index']][
                        'Document Name'].to_list()
                    for doc_dir_to_index in document_dirs:
                        file_name = os.path.splitext(os.path.basename(doc_dir_to_index))[0]
                        if mode == 'Create':
                            self.indexing_util.index_document(
                                document_directory=doc_dir_to_index,
                                index_directory=os.path.join(self.indices_dir, file_name),
                                chunk_size=chunk_size,
                                embedding_model=embedding_model
                            )
                            st.success(f"Document index {file_name} saved! Refreshing page now.")
                        elif mode == 'Delete':
                            self.indexing_util.delete_document(
                                selected_directory=os.path.join(self.indices_dir, file_name)
                            )
                            self.indexing_util.delete_document(
                                selected_directory=os.path.join(self.document_dir, file_name)
                            )
                            st.error(f"Document index {file_name} deleted! Refreshing page now.")

            time.sleep(2)
            st.experimental_rerun()

    def render_document_explore_page(self):
        """
        """
        colored_header(
            label="Explore Document",
            description="Select document, generate index!",
            color_name="blue-green-70",
        )
        with st.form('explore_document'):
            st.markdown("<h6>Select Document:</h6>", unsafe_allow_html=True)
            documents_df = self.indexing_util.get_available_documents(
                document_dir=self.document_dir, indices_dir=self.indices_dir
            )
            documents_df = documents_df.rename(columns={'Select Index': 'Select Document'})
            documents_df['Creation Date'] = pd.to_datetime(documents_df['Creation Date'])
            documents_df = documents_df.sort_values(by='Creation Date', ascending=False)
            selected_documents_df = st.data_editor(documents_df, hide_index=True, use_container_width=True)
            submitted = st.form_submit_button("Explore!", type="primary")
            if submitted:
                selected_docs_dir_paths = selected_documents_df[
                    selected_documents_df['Select Document']
                ]['Document Name'].to_list()
                data = []
                for selected_doc_dir_path in selected_docs_dir_paths:
                    filename = selected_doc_dir_path.split('/')[-1] + '.txt'
                    filepath = os.path.join(selected_doc_dir_path, filename)
                    with open(filepath, 'r') as f:
                        text = f.read()
                        data.append({"filename": filename, "text": text})

                with st.expander("Text", expanded=False):
                    for doc in data:
                        st.markdown(f"<h6>File: {doc['filename']}</h6>", unsafe_allow_html=True)
                        st.markdown(f"<p>{' '.join(doc['text'].split())}</p>", unsafe_allow_html=True)
                with st.expander("Word Clouds", expanded=False):
                    for doc in data:
                        st.markdown(f"<h6>File: {doc['filename']}</h6>", unsafe_allow_html=True)
                        plt = generate_wordcloud(text=doc['text'], background_color='black', colormap='Pastel1')
                        st.pyplot(plt)

                for index, doc in enumerate(data):
                    doc = self.nlp(' '.join(doc['text'].split()))
                    visualize_ner(doc, labels=self.ner_labels, show_table=False, key=f"doc_{index}")

    def render_qa_page(self, temperature=None, max_tokens=None, model_name=None, chain_type=None):
        """
        """

        def _gen_summary(t):
            keywords = ["summarize", "summary"]
            for keyword in keywords:
                if keyword in t.lower():
                    return True
            return False

        colored_header(
            label="Q & A",
            description="Select index, ask questions!",
            color_name="red-70",
        )
        st.info(f"\n\ntemperature: {temperature}, max_tokens: {max_tokens}, model_name: {model_name}")
        with st.container():
            # enable_audio = st.checkbox("Enable TTS")
            indices_df = self.indexing_util.get_available_indices(indices_dir=self.indices_dir)
            selected_index_path = selectbox(
                "Select Index:", options=indices_df['Index Name'].to_list(),
                no_selection_label="<select index>", label_visibility="collapsed"
            )

            if selected_index_path is None:
                st.error("Select index first!")
                return

        if selected_index_path is not None:
            chat_history_filepath = os.path.join(
                self.chat_history_dir, f"{os.path.basename(selected_index_path)}.pickle"
            )

            # Initialize chat history
            _chat_history = []
            if selected_index_path not in st.session_state:

                # check if chat history is available locally, if yes; load the chat history
                if os.path.exists(chat_history_filepath):
                    log_debug(f"Loading chat history from local file: {chat_history_filepath}")
                    with open(chat_history_filepath, 'rb') as f:
                        st.session_state[selected_index_path] = pickle.load(f)
                else:
                    st.session_state[selected_index_path] = {'messages': []}

            with st.spinner('thinking...'):

                # Display chat messages from history on app rerun
                for message_item in st.session_state[selected_index_path]['messages']:
                    with st.chat_message(message_item["role"]):
                        st.markdown(message_item["content"])

                # React to user input
                if prompt := st.chat_input(f"Start asking questions to '{selected_index_path[:25]}...' index!"):

                    # Add user message to chat history
                    st.session_state[selected_index_path]['messages'].append({
                        "role": "user", "content": prompt
                    })

                    # Display user message in chat message container
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    # Define summarization mechanism here
                    summarization_chain = self.summary_util.initialize_summarization_chain(
                        temperature=temperature, max_tokens=max_tokens, chain_type=chain_type
                    )

                    # Define Q/A mechanism here
                    qa_chain = self.ask_util.prepare_qa_chain(
                        index_directory=selected_index_path,
                        temperature=temperature,
                        model_name=model_name,
                        max_tokens=max_tokens
                    )
                    if _gen_summary(prompt):
                        # If the prompt is asking to summarize
                        log_info("Summarization")
                        doc_filepath = os.path.join(
                            self.document_dir, os.path.basename(selected_index_path),
                            f"{os.path.basename(selected_index_path)}.txt"
                        )
                        with open(doc_filepath, 'r') as f:
                            text = f.read()
                        answer, answer_meta, chat_history = self.summary_util.summarize(
                            chain=summarization_chain, text=text, question=prompt,
                            chat_history=_chat_history
                        )
                        # Display assistant response in chat message container
                        with st.chat_message("assistant"):
                            st.markdown(answer)
                            st.info(answer_meta)
                    else:
                        # Other Q/A questions
                        log_info("QA")
                        answer, answer_meta, chat_history = self.ask_util.ask_question(
                            question=prompt, qa_chain=qa_chain, chat_history=_chat_history
                        )
                        # Display assistant response in chat message container
                        with st.chat_message("assistant"):
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
                            # if enable_audio:
                            #     wav = self.tts.tts(full_response)
                            #     wav_array = np.array(wav)
                            #     sample_rate = 22500
                            #     st.audio(wav_array, format='audio/wav', sample_rate=sample_rate)

                    _chat_history.extend(chat_history)

                    # Add assistant response to chat history
                    st.session_state[selected_index_path]['messages'].append({
                        "role": "assistant", "content": answer
                    })
                    # Save conversation to local file
                    log_debug(f"Saving chat history to local file: {chat_history_filepath}")
                    with open(chat_history_filepath, 'wb') as f:
                        pickle.dump(st.session_state[selected_index_path], f)

    def render_test_page(self):
        """

        """
        # with st.form("process_text"):
        #     text = st.text_area("Enter text:")
        #     submit = st.form_submit_button('Process')
        #     if submit:
        #         wav = self.tts.tts(text)
        #         wav_array = np.array(wav)
        #         sample_rate = 22500
        #         st.audio(wav_array, format='audio/wav', sample_rate=sample_rate)


def main():
    app_version = "0.0.3"
    st.set_page_config(
        page_title="VerbalVista",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/spate141/VerbalVista',
            'Report a bug': "https://github.com/spate141/VerbalVista",
            'About': "### Welcome to VerbalVista!\nBuilt by Snehal Patel."
        }
    )
    st.sidebar.markdown(
        f"""
        <center>
        <a href="https://github.com/spate141/VerbalVista"><img src="https://i.ibb.co/6FQPs5C/verbal-vista-blue-transparent.png" width="70%" height="70%"></a>
        </br>
        </br>
        <h5>Version: {app_version}</h5>
        </center>
        <i class="fa-regular fa-v fa-flip"></i>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.markdown("<center><h4><b>OpenAI API Key</b></h5></center>", unsafe_allow_html=True)
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key:", key="chatbot_api_key", type="password", label_visibility="collapsed"
    )
    st.sidebar.markdown("<center><h4><b>Select Function</b></h5></center>", unsafe_allow_html=True)
    page = st.sidebar.selectbox(
        "Select function:", [
            "Media Processing",
            "Explore Document",
            "Manage Index",
            "Q & A",
            "test"
        ], label_visibility="collapsed"
    )

    document_dir = 'data/documents/'
    tmp_audio_dir = 'data/tmp_audio_dir/'
    indices_dir = 'data/indices/'
    chat_history_dir = 'data/chat_history/'

    if not os.environ.get("OPENAI_API_KEY", None) and not openai_api_key:
        # if both env variable and explicit key is not set
        st.error("OpenAI API key not found!")
        log_error("No OpenAI key found!")
    elif not os.environ.get("OPENAI_API_KEY", None) and openai_api_key:
        # if env variable is not set but user provide explicit key
        os.environ['OPENAI_API_KEY'] = openai_api_key
        log_info(f"No OpenAI key found in ENV, User provided key.")
    elif os.environ.get("OPENAI_API_KEY", None) and openai_api_key:
        # if both env variable and explicit keys are provided
        os.environ['OPENAI_API_KEY'] = openai_api_key
        log_info(f"OpenAI key found in ENV & User provided key.")

    vv = VerbalVista(
        document_dir=document_dir, indices_dir=indices_dir,
        tmp_audio_dir=tmp_audio_dir, chat_history_dir=chat_history_dir
    )
    if page == "Media Processing":
        vv.render_media_processing_page()
    elif page == "Manage Index":
        vv.render_manage_index_page()
    elif page == "Q & A":
        with st.sidebar:
            temperature = st.number_input("Temperature", value=0.5, min_value=0.0, max_value=1.0)
            max_tokens = st.number_input("Max Tokens", value=512, min_value=0, max_value=4000)
            model_name = st.selectbox("Model Name", ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"], index=0)
            summ_chain_type = st.selectbox("Chain type", index=1, options=["stuff", "map_reduce", "refine"])
        vv.render_qa_page(temperature=temperature, max_tokens=max_tokens, model_name=model_name, chain_type=summ_chain_type)
    elif page == "Explore Document":
        vv.render_document_explore_page()
    elif page == 'test':
        vv.render_test_page()


if __name__ == '__main__':
    main()
