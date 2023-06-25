import os
import time
import streamlit as st
from streamlit_extras.colored_header import colored_header
from streamlit_extras.no_default_selectbox import selectbox
from utils.audio_transcribe import WhisperAudioTranscribe
from utils.generate_vector_index import VectorIndex, load_index
from utils.logging_module import log_info, log_debug, log_error
from utils.document_reader import parse_docx, parse_pdf, parse_txt, write_text_to_file, text_to_docs
from langchain.callbacks import get_openai_callback


class VerbalVista:

    def __init__(self):
        self.whisper = WhisperAudioTranscribe()
        self.vector_index = VectorIndex()

    def render_media_processing_page(self, tmp_document_dir: str = None, tmp_audio_dir: str = None):
        """
        :param tmp_document_dir:
        :param tmp_audio_dir:
        :return:
        """
        supported_formats = ['m4a', 'mp3', 'wav', 'docx', 'pdf', 'txt']
        colored_header(
            label="Process Media",
            description=f"Supported formats: {supported_formats}",
            color_name="violet-70",
        )
        with st.form('docs_processing'):
            uploaded_file = st.file_uploader(
                "Upload file:", type=supported_formats, accept_multiple_files=False,
                label_visibility="collapsed"
            )
            _, col, _ = st.columns([2, 8, 2])
            with col:
                st.markdown("<center><h6><u>Available Documents</u></h6></center>", unsafe_allow_html=True)
                transcript_df = self.vector_index.get_available_documents(tmp_document_dir=tmp_document_dir)
                st.dataframe(
                    transcript_df, hide_index=True, use_container_width=True,
                    column_order=['Index Status', 'Document Name', 'Creation Date']
                )
            submitted = st.form_submit_button("Submit")
            if submitted:
                if uploaded_file is not None:

                    # each conditions process different kind of media and returns media content as string of text
                    if uploaded_file.name.endswith(('.m4a', '.mp3', '.wav')):
                        with st.spinner('Processing audio. Please wait.'):
                            process_audio_bar = st.progress(0, text="Processing...")
                            # Save the uploaded file to the specified directory
                            tmp_audio_save_path = os.path.join(tmp_audio_dir, uploaded_file.name)
                            log_debug(f"tmp_save_path: {tmp_audio_save_path}")
                            with open(tmp_audio_save_path, "wb") as f:
                                f.write(uploaded_file.getvalue())
                            # Generate audio chunks
                            audio_chunks_files, file_size_mb, file_duration_in_ms = self.whisper.generate_audio_chunks(
                                audio_filepath=tmp_audio_save_path, max_audio_size=25, tmp_dir=tmp_audio_dir,
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
                        self.whisper.remove_temp_files(tmp_audio_dir)

                    elif uploaded_file.name.endswith(".pdf"):
                        with st.spinner('Processing pdf file. Please wait.'):
                            full_document = parse_pdf(uploaded_file)

                    elif uploaded_file.name.endswith(".docx"):
                        with st.spinner('Processing word file. Please wait.'):
                            full_document = parse_docx(uploaded_file)

                    elif uploaded_file.name.endswith(".txt"):
                        with st.spinner('Processing text file. Please wait.'):
                            full_document = parse_txt(uploaded_file)

                    # Write document to a file
                    st.markdown("#### Document snippet:")
                    st.caption(full_document[:110] + '...')
                    uploaded_file_name = uploaded_file.name.replace('.', '_')
                    tmp_document_save_path = write_text_to_file(
                        uploaded_file_name=uploaded_file_name, tmp_document_dir=tmp_document_dir,
                        full_document=full_document
                    )
                    st.success(f"Document saved: {tmp_document_save_path}")
                    time.sleep(2)
                    st.experimental_rerun()

    def render_create_index_page(self, tmp_document_dir: str = None, tmp_indices_dir: str = None):
        """

        :param tmp_document_dir:
        :param tmp_indices_dir:
        :return:
        """
        colored_header(
            label="Create Index",
            description="Select document, generate index!",
            color_name="blue-green-70",
        )
        with st.form('create_index'):
            col1, col2, col3 = st.columns([6, 0.1, 5], gap='small')
            with col1:
                st.markdown("<center><h6><u>Available Documents</u></h6></center>", unsafe_allow_html=True)
                transcript_df = self.vector_index.get_available_documents(tmp_document_dir=tmp_document_dir)
                selected_transcripts_df = st.data_editor(transcript_df, hide_index=True, use_container_width=True)
            with col3:
                st.markdown("<center><h6><u>Available Indices</u></h6></center>", unsafe_allow_html=True)
                indices_df = self.vector_index.get_available_indices(tmp_indices_dir=tmp_indices_dir)
                st.dataframe(indices_df, hide_index=True)

            st.markdown("<center><h6><u>Modify LangChain PromptHelper Parameters</u></h6></center>", unsafe_allow_html=True)
            cols = st.columns(4)
            with cols[0]:
                context_window = st.number_input("context_window:", value=3900)
            with cols[1]:
                num_outputs = st.number_input("num_outputs:", value=512)
            with cols[2]:
                chunk_overlap_ratio = st.number_input("chunk_overlap_ratio:", value=0.1)
            with cols[3]:
                chunk_size_limit = st.number_input("chunk_size_limit:", value=600)

            submitted = st.form_submit_button("Create Index")
            if submitted:
                _, c, _ = st.columns([2, 5, 2])
                with c:
                    with st.spinner('Indexing document. Please wait.'):
                        document_dirs = selected_transcripts_df[selected_transcripts_df['Create Index']][
                            'Document Name'].to_list()
                        for doc_dir_to_index in document_dirs:
                            file_name = os.path.splitext(os.path.basename(doc_dir_to_index))[0]
                            self.vector_index.index_document(
                                document_directory=doc_dir_to_index,
                                index_directory=os.path.join(tmp_indices_dir, file_name),
                                context_window=context_window, num_outputs=num_outputs,
                                chunk_overlap_ratio=chunk_overlap_ratio,
                                chunk_size_limit=chunk_size_limit
                            )
                    st.success(f"Document index {file_name} saved! Refreshing page now.")
                time.sleep(2)
                st.experimental_rerun()

    def render_qa_page(self, tmp_indices_dir: str = None):
        """

        :param tmp_indices_dir:
        :return:
        """
        colored_header(
            label="Q & A",
            description="Select index, ask questions!",
            color_name="red-70",
        )
        with st.form('index_selection'):
            _, col, _ = st.columns([2, 5, 2])
            with col:
                st.markdown("<center><h6><u>Select Index for Q & A</u></h6></center>", unsafe_allow_html=True)
                indices_df = self.vector_index.get_available_indices(tmp_indices_dir=tmp_indices_dir)
                selected_index_path = selectbox(
                    "Select Index:", options=indices_df['Index Name'].to_list(), no_selection_label="<select index>",
                    label_visibility="collapsed"
                )
                st.markdown("<center><h6><u>Enter Query</u></h6></center>", unsafe_allow_html=True)
                question = st.text_input("Question:", label_visibility="collapsed")

                _, col, _ = st.columns([100, 25, 100])
                with col:
                    # load the index and allow user to ask question at a same time
                    submitted = st.form_submit_button("Ask!", type='primary')

                if submitted:
                    start = time.time()
                    if selected_index_path is None:
                        st.error("Error: Select index first!")
                    else:
                        st.info(f"Selected index: {selected_index_path}")
                        with st.spinner('🤖 thinking...'):
                            query_engine = load_index(selected_index_path)
                            with get_openai_callback() as cb:
                                response = query_engine.query(question)
                            total_time = round(time.time() - start, 2)
                            st.write(response.response)
                            query_meta = f"""
                            - Total Tokens Used: {cb.total_tokens}
                              - Prompt Tokens: {cb.prompt_tokens}
                              - Completion Tokens: {cb.completion_tokens}
                            - Successful Requests: {cb.successful_requests}
                            - Total Cost (USD): ${round(cb.total_cost, 4)}
                            - Total Time (Seconds): {total_time}
                            """
                            st.success(query_meta)


def main():

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
        """
        <center><a href="https://github.com/spate141/VerbalVista"><img src="https://github-production-user-asset-6210df.s3.amazonaws.com/10580847/248083735-15c326d9-67df-4fb2-b50a-c7684f45bacb.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20230623%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230623T043013Z&X-Amz-Expires=300&X-Amz-Signature=935eb36a4695ea2b21e7f93e8958de3058fcda1f355b4498ab6a5e2dd2f1d7cc&X-Amz-SignedHeaders=host&actor_id=10580847&key_id=0&repo_id=656493437" width="70%" height="70%"></a></center>
        </br>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.markdown("<center><h4><b>Select Function</b></h5></center>", unsafe_allow_html=True)
    page = st.sidebar.selectbox(
        "Select function:", [
            "Media Processing",
            "Create Index",
            "Q & A"
        ], label_visibility="collapsed"
    )

    vv = VerbalVista()
    tmp_document_dir = 'documents/'
    tmp_audio_dir = 'tmp_audio_dir/'
    tmp_indices_dir = 'indices/'
    if page == "Media Processing":
        vv.render_media_processing_page(tmp_document_dir=tmp_document_dir, tmp_audio_dir=tmp_audio_dir)
    elif page == "Create Index":
        vv.render_create_index_page(tmp_document_dir=tmp_document_dir, tmp_indices_dir=tmp_indices_dir)
    elif page == "Q & A":
        vv.render_qa_page(tmp_indices_dir=tmp_indices_dir)


if __name__ == '__main__':
    main()
