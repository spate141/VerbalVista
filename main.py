import os
import time
import pickle
import pandas as pd
import streamlit as st
from streamlit_extras.colored_header import colored_header
from streamlit_extras.no_default_selectbox import selectbox
from utils.audio_transcribe import WhisperAudioTranscribe
from utils.indexing_util import MyIndex
from utils.logging_module import log_info, log_debug, log_error
from utils.document_parser import parse_docx, parse_pdf, parse_txt, write_text_to_file, extract_text_from_url
from utils.generate_wordcloud import generate_wordcloud


class VerbalVista:

    def __init__(self, tmp_document_dir: str = None, tmp_audio_dir: str = None, tmp_indices_dir: str = None, tmp_chat_history_dir: str = None):
        self.whisper = WhisperAudioTranscribe()
        self.indexing_util = MyIndex()

        _ = [self.create_directory(d) for d in [tmp_document_dir, tmp_indices_dir, tmp_audio_dir, tmp_chat_history_dir]]
        self.tmp_document_dir = tmp_document_dir
        self.tmp_indices_dir = tmp_indices_dir
        self.tmp_audio_dir = tmp_audio_dir
        self.tmp_chat_history_dir = tmp_chat_history_dir

    @staticmethod
    def create_directory(directory_path):
        """

        :param directory_path:
        :return:
        """
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            log_info(f"Directory '{directory_path}' created successfully.")
        else:
            log_info(f"Directory '{directory_path}' already exists.")

    def render_media_processing_page(self):
        """
        """
        supported_formats = ['m4a', 'mp3', 'wav', 'docx', 'pdf', 'txt']
        colored_header(
            label="Process Media",
            description=f"Supported formats: {supported_formats}",
            color_name="violet-70",
        )
        with st.form('docs_processing'):

            col1, col2, col3 = st.columns([14, 8, 8])
            with col1:
                st.markdown("<h6>Process local files:</h6>", unsafe_allow_html=True)
                uploaded_file = st.file_uploader(
                    "Upload file:", type=supported_formats, accept_multiple_files=False,
                    label_visibility="collapsed"
                )
            with col2:
                st.markdown("<h6>Extract text from URL:</h6>", unsafe_allow_html=True)
                url = st.text_area("Enter URL:", placeholder='https://YOUR_URL', label_visibility="collapsed")
                url = None if len(url) == 0 else url
            with col3:
                st.markdown("<h6>Copy/Paste text:</h6>", unsafe_allow_html=True)
                text = st.text_area("Paste text:", placeholder='YOUR TEXT', label_visibility="collapsed")
                text = None if len(text) == 0 else text

            submitted = st.form_submit_button("Submit")
            if submitted:
                full_document = ''

                if uploaded_file is not None:
                    log_debug('Processing uploaded file!')
                    if uploaded_file.name.endswith(('.m4a', '.mp3', '.wav')):
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

                        log_info(f"Removing tmp audio files")
                        self.whisper.remove_temp_files(self.tmp_audio_dir)

                    elif uploaded_file.name.endswith(".pdf"):
                        with st.spinner('Processing pdf file. Please wait.'):
                            full_document = parse_pdf(uploaded_file)

                    elif uploaded_file.name.endswith(".docx"):
                        with st.spinner('Processing word file. Please wait.'):
                            full_document = parse_docx(uploaded_file)

                    elif uploaded_file.name.endswith(".txt"):
                        with st.spinner('Processing text file. Please wait.'):
                            full_document = parse_txt(uploaded_file)

                    uploaded_file_name = uploaded_file.name.replace('.', '_').replace(' ', '_')

                elif url is not None:
                    log_debug('Processing URL!')
                    full_document = extract_text_from_url(url)
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
                    tmp_document_save_path = write_text_to_file(
                        uploaded_file_name=uploaded_file_name,
                        tmp_document_dir=self.tmp_document_dir,
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
        with st.form('manage_index'):
            st.markdown(
                "<center><h6><u>Modify LangChain PromptHelper Parameters</u></h6></center>", unsafe_allow_html=True
            )
            cols = st.columns(3)
            with cols[0]:
                model_name = st.selectbox("model_name:", options=[
                    "gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-16k"
                ], index=0)
            with cols[1]:
                temperature = st.number_input("temperature:", value=0.7)
            with cols[2]:
                chunk_size_limit = st.number_input("chunk_size_limit:", value=600)
            st.markdown("</br>", unsafe_allow_html=True)

            cols = st.columns(3)
            with cols[0]:
                context_window = st.number_input("context_window:", value=3900)
            with cols[1]:
                num_outputs = st.number_input("num_outputs:", value=512)
            with cols[2]:
                chunk_overlap_ratio = st.number_input("chunk_overlap_ratio:", value=0.1)
            st.markdown("</br>", unsafe_allow_html=True)

            cols = st.columns([6, 0.1, 5], gap='small')
            with cols[0]:
                st.markdown("<center><h6><u>Available Documents</u></h6></center>", unsafe_allow_html=True)
                documents_df = self.indexing_util.get_available_documents(tmp_document_dir=self.tmp_document_dir)
                documents_df['Creation Date'] = pd.to_datetime(documents_df['Creation Date'])
                documents_df = documents_df.sort_values(by='Creation Date', ascending=False)
                selected_documents_df = st.data_editor(documents_df, hide_index=True, use_container_width=True)
            with cols[2]:
                st.markdown("<center><h6><u>Available Indices</u></h6></center>", unsafe_allow_html=True)
                indices_df = self.indexing_util.get_available_indices(tmp_indices_dir=self.tmp_indices_dir)
                indices_df['Creation Date'] = pd.to_datetime(indices_df['Creation Date'])
                indices_df = indices_df.sort_values(by='Creation Date', ascending=False)
                st.dataframe(indices_df, hide_index=True)

            submitted = st.form_submit_button("Create Index")
            if submitted:
                _, c, _ = st.columns([2, 5, 2])
                with c:
                    with st.spinner('Indexing document. Please wait.'):
                        document_dirs = selected_documents_df[selected_documents_df['Create Index']][
                            'Document Name'].to_list()
                        for doc_dir_to_index in document_dirs:
                            file_name = os.path.splitext(os.path.basename(doc_dir_to_index))[0]
                            self.indexing_util.index_document(
                                document_directory=doc_dir_to_index,
                                index_directory=os.path.join(self.tmp_indices_dir, file_name),
                                context_window=context_window, num_outputs=num_outputs,
                                chunk_overlap_ratio=chunk_overlap_ratio,
                                chunk_size_limit=chunk_size_limit,
                                temperature=temperature,
                                model_name=model_name
                            )
                    st.success(f"Document index {file_name} saved! Refreshing page now.")
                time.sleep(2)
                st.experimental_rerun()

    def render_qa_page(self):
        """
        """
        colored_header(
            label="Q & A",
            description="Select index, ask questions!",
            color_name="red-70",
        )
        st.markdown("<center><h6><u>Select Index for Q & A</u></h6></center>", unsafe_allow_html=True)
        indices_df = self.indexing_util.get_available_indices(tmp_indices_dir=self.tmp_indices_dir)
        selected_index_path = selectbox(
            "Select Index:", options=indices_df['Index Name'].to_list(),
            no_selection_label="<select index>", label_visibility="collapsed"
        )

        if selected_index_path is None:
            st.error("Error: Select index first!")
        else:
            st.info(f"Selected index: {selected_index_path}")
            chat_history_filepath = os.path.join(
                self.tmp_chat_history_dir, f"{os.path.basename(selected_index_path)}.pickle"
            )

            # Initialize chat history
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

                    # Define Q/A mechanism here
                    response, response_meta = self.indexing_util.generate_answer(
                        prompt=prompt, selected_index_path=selected_index_path
                    )

                    # Display assistant response in chat message container
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        full_response = ""

                        # Simulate stream of response with milliseconds delay
                        for chunk in response.response.split():
                            full_response += chunk + " "
                            time.sleep(0.03)

                            # Add a blinking cursor to simulate typing
                            message_placeholder.markdown(full_response + "▌")

                        # Display full message at the end with other stuff you want to show like `response_meta`.
                        message_placeholder.markdown(full_response)
                        st.info(response_meta)

                    # Add assistant response to chat history
                    st.session_state[selected_index_path]['messages'].append({
                        "role": "assistant", "content": response
                    })
                    # Save conversation to local file
                    log_debug(f"Saving chat history to local file: {chat_history_filepath}")
                    with open(chat_history_filepath, 'wb') as f:
                        pickle.dump(st.session_state[selected_index_path], f)

    def render_document_explore_page(self):
        """
        """
        colored_header(
            label="Explore Document",
            description="Select document, generate index!",
            color_name="blue-green-70",
        )
        with st.form('explore_document'):
            st.markdown("<center><h6><u>Select Document</u></h6></center>", unsafe_allow_html=True)
            documents_df = self.indexing_util.get_available_documents(tmp_document_dir=self.tmp_document_dir)
            documents_df = documents_df.rename(columns={'Create Index': 'Select Document'})
            documents_df['Creation Date'] = pd.to_datetime(documents_df['Creation Date'])
            documents_df = documents_df.sort_values(by='Creation Date', ascending=False)
            selected_documents_df = st.data_editor(documents_df, hide_index=True, use_container_width=False)
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
                with st.expander("Word Clouds"):
                    for doc in data:
                        st.markdown(f"<h6>File: {doc['filename']}</h6>", unsafe_allow_html=True)
                        plt = generate_wordcloud(text=doc['text'], background_color='black', colormap='Pastel1')
                        st.pyplot(plt)


def main():
    VERSION = 0.2
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
        <h5>Version: {VERSION}</h5>
        </center>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.markdown("<center><h4><b>Select Function</b></h5></center>", unsafe_allow_html=True)
    page = st.sidebar.selectbox(
        "Select function:", [
            "Media Processing",
            "Explore Document",
            "Manage Index",
            "Q & A"
        ], label_visibility="collapsed"
    )

    tmp_document_dir = 'documents/'
    tmp_audio_dir = 'tmp_audio_dir/'
    tmp_indices_dir = 'indices/'
    tmp_chat_history_dir = 'chat_history/'

    vv = VerbalVista(
        tmp_document_dir=tmp_document_dir, tmp_indices_dir=tmp_indices_dir,
        tmp_audio_dir=tmp_audio_dir, tmp_chat_history_dir=tmp_chat_history_dir
    )
    if page == "Media Processing":
        vv.render_media_processing_page()
    elif page == "Manage Index":
        vv.render_manage_index_page()
    elif page == "Q & A":
        vv.render_qa_page()
    elif page == "Explore Document":
        vv.render_document_explore_page()


if __name__ == '__main__':
    main()
