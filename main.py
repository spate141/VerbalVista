import os
import time
import streamlit as st
from utils.audio_transcribe import WhisperAudioTranscribe
from utils.generate_vector_index import VectorIndex
from utils.logging_module import log_info, log_debug, log_error


class VerbalVista:

    def __init__(self):
        self.whisper = WhisperAudioTranscribe()
        self.vector_index = VectorIndex()

    def render_audio_transcribe_page(self, tmp_transcript_dir: str = 'transcripts/', tmp_audio_dir: str = 'tmp_dir/'):
        """
        :param tmp_transcript_dir:
        :param tmp_audio_dir:
        :return:
        """
        with st.form('audio_transcribe'):
            st.markdown("#### Upload audio file:")
            uploaded_file = st.file_uploader(
                "Upload an audio file:", type=['m4a', 'mp3', 'wav'], accept_multiple_files=False,
                label_visibility="collapsed"
            )

            submitted = st.form_submit_button("Submit")
            if submitted:
                if uploaded_file is not None:
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
                    full_audio_transcript = []
                    with st.spinner('Transcribing audio. Please wait.'):
                        transcribe_audio_bar = st.progress(0, text="Transcribing...")
                        total_chunks = len(audio_chunks_files)
                        pct_cmp = [i / total_chunks for i in range(1, total_chunks + 1)]
                        for index, i in enumerate(audio_chunks_files):
                            transcript = self.whisper.transcribe_audio(i)
                            full_audio_transcript.append(transcript)
                            transcribe_audio_bar.progress(
                                pct_cmp[index - 1], f'Audio transcribed: {round(time.time()-start,2)} sec'
                            )
                        full_audio_transcript = ' '.join(full_audio_transcript)

                    # Write transcript to a file
                    st.markdown("#### Transcript snippet:")
                    st.caption(full_audio_transcript[:110] + '...')
                    tmp_transcript_save_path = self.whisper.write_transcript_to_file(
                        uploaded_file_name=uploaded_file.name, tmp_transcript_dir=tmp_transcript_dir,
                        full_audio_transcript=full_audio_transcript
                    )
                    st.success(f"Audio transcript saved: {tmp_transcript_save_path}")
                    log_info(f"Removing tmp audio files")
                    self.whisper.remove_temp_files(tmp_audio_dir)

    def render_create_index_page(self, tmp_transcript_dir: str = 'transcripts/', tmp_indices_dir: str = 'indices/'):
        """

        :param tmp_transcript_dir:
        :param tmp_indices_dir:
        :return:
        """
        with st.form('create_index'):
            st.markdown("#### Create Index:")
            col1, col2, col3 = st.columns([6, 1, 5], gap='small')
            with col1:
                st.markdown("###### Available Transcripts:")
                transcript_df = self.vector_index.get_available_transcripts(tmp_transcript_dir=tmp_transcript_dir)
                selected_transcripts_df = st.data_editor(transcript_df, hide_index=True, use_container_width=True)
            with col3:
                st.markdown("###### Available Indices:")
                indices_df = self.vector_index.get_available_indices(tmp_indices_dir=tmp_indices_dir)
                st.dataframe(indices_df, hide_index=True)

            st.markdown("###### LangChain PromptHelper Parameters:")
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
                    with st.spinner('Indexing transcript. Please wait.'):
                        transcript_dirs = selected_transcripts_df[selected_transcripts_df['Create Index']][
                            'Transcript Name'].to_list()
                        for transcript_dir_to_index in transcript_dirs:
                            file_name = os.path.splitext(os.path.basename(transcript_dir_to_index))[0]
                            self.vector_index.index_document(
                                transcript_directory=transcript_dir_to_index,
                                index_directory=os.path.join(tmp_indices_dir, file_name),
                                context_window=context_window, num_outputs=num_outputs,
                                chunk_overlap_ratio=chunk_overlap_ratio,
                                chunk_size_limit=chunk_size_limit
                            )
                    st.success(f"Transcript index {file_name} saved! Refreshing page now.")
                time.sleep(2)
                st.experimental_rerun()

    @staticmethod
    def render_qa_page():
        with st.form('qa'):
            st.markdown("#### Q & A:")
            submitted = st.form_submit_button("Submit")
            if submitted:
                pass


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
    vv = VerbalVista()

    st.sidebar.title("Verbal Vista")
    page = st.sidebar.selectbox("Select function:", ["Transcribe Audio", "Create Index", "Q & A"])
    if page == "Transcribe Audio":
        vv.render_audio_transcribe_page()
    elif page == "Create Index":
        vv.render_create_index_page()
    elif page == "Q & A":
        vv.render_qa_page()


if __name__ == '__main__':
    main()
