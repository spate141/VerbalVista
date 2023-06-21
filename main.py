import os
import time
import streamlit as st
from utils.audio_transcribe import WhisperAudioTranscribe
from utils.logging_module import log_info, log_debug, log_error


class VerbalVista:

    def __init__(self):
        self.whisper = WhisperAudioTranscribe()

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

                    # Save transcript
                    st.markdown("#### Transcript snippet:")
                    st.caption(full_audio_transcript[:110] + '...')
                    transcript_file_name = os.path.splitext(uploaded_file.name)[0] + '.txt'
                    tmp_transcript_save_path = os.path.join(tmp_transcript_dir, transcript_file_name)
                    with open(tmp_transcript_save_path, 'w') as f:
                        f.write(full_audio_transcript)
                    st.success(f"Audio transcript saved: {tmp_transcript_save_path}")
                    log_info(f"Removing tmp audio files")
                    self.whisper.remove_temp_files(tmp_audio_dir)


def main():

    vv = VerbalVista()

    st.sidebar.title("Verbal Vista")
    page = st.sidebar.selectbox("Select function:", ["Create Index", "Q & A"])
    if page == "Create Index":
        vv.render_audio_transcribe_page()
    elif page == "Q & A":
        pass


if __name__ == '__main__':
    main()
