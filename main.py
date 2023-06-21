import os
import time
import openai
import streamlit as st
from pydub import AudioSegment
from pydub.utils import mediainfo
from utils.logging_module import log_info, log_debug, log_error

openai.api_key = os.getenv("OPENAI_API_KEY")


class VerbalVista:

    def __init__(self):
        pass

    @staticmethod
    def load_audio_data(filepath):
        """

        :param filepath:
        :return:
        """
        # load audio
        extension = os.path.splitext(filepath)[-1][1:]
        audio = AudioSegment.from_file(filepath, format=extension)
        log_info(f"Audio file read into memory: {filepath}")

        # get the original audio file size in MB and duration in ms
        file_size_mb = (float(mediainfo(filepath)['size']) / 1024) / 1024
        file_duration_in_ms = float(mediainfo(filepath)['duration']) * 1000
        log_debug(f"Audio meta: {file_size_mb} MBs, {file_duration_in_ms} ms")
        return {
            "audio": audio,
            "file_size_mb": file_size_mb,
            "file_duration_in_ms": file_duration_in_ms
        }

    @staticmethod
    def create_a_file(directory, filename):
        """

        :param directory:
        :param filename:
        :return:
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, filename)
        return file_path

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

    @staticmethod
    def split_sizes_into_chunks(actual_size: float = None, max_size: int = None) -> list:
        """
        Given an actual_size in MB; this function will return list of MBs that are <= max_size
        :param actual_size:
        :param max_size:
        :return:
        """
        chunks = []
        remaining = actual_size
        while remaining > 0:
            chunk_size = min(max_size, remaining)
            chunks.append(chunk_size)
            remaining -= chunk_size
        return chunks

    def get_audio_chunks_duration(
            self, file_size_mb: float = None, file_duration_in_ms: float = None, max_audio_size: int = 25
    ):
        """
        This will return list of audio durations in milliseconds that is less than or equal to the
        max_audio_size in MB.
        :param file_size_mb:
        :param file_duration_in_ms:
        :param max_audio_size:
        :return:
        """
        # split the original audio file into appropriate chunks of size <= max_audio_size
        file_chunks_sizes_mb = self.split_sizes_into_chunks(actual_size=file_size_mb, max_size=max_audio_size)

        # find the audio duration of each chunk based on the audio chunk size in MB
        file_chunks_durations_ms = [(i * file_duration_in_ms) / file_size_mb for i in file_chunks_sizes_mb]
        assert sum(file_chunks_durations_ms) == file_duration_in_ms
        return file_chunks_durations_ms

    def generate_audio_chunks(
            self, audio_filepath: str = None, max_audio_size: int = 25, tmp_dir: str = 'tmp_dir/',
            process_bar=None
    ):
        """

        :param audio_filepath:
        :param max_audio_size:
        :param tmp_dir:
        :param process_bar:
        :return:
        """
        start = time.time()
        audio_meta = self.load_audio_data(audio_filepath)
        audio = audio_meta['audio']
        file_size_mb = audio_meta['file_size_mb']
        file_duration_in_ms = audio_meta['file_duration_in_ms']

        file_chunks_durations_ms = self.get_audio_chunks_duration(
            file_size_mb=file_size_mb,
            file_duration_in_ms=file_duration_in_ms,
            max_audio_size=max_audio_size
        )
        total_chunks = len(file_chunks_durations_ms)
        pct_cmp = [i / total_chunks for i in range(1, total_chunks + 1)]
        audio_chunks_files = []
        for index, chunk_duration_ms in enumerate(file_chunks_durations_ms, 1):
            output_file = self.create_a_file(tmp_dir, f"tmp_audio_chunk_{index}.mp3")
            log_debug(f"Audio chunk created: {output_file}")
            audio_chunk = audio[:chunk_duration_ms]
            audio_chunk.export(output_file, format="mp3")
            audio = audio[chunk_duration_ms:]
            audio_chunks_files.append(output_file)
            process_bar.progress(pct_cmp[index - 1], f'Audio processed: {round(time.time()-start,2)} sec')
        return audio_chunks_files, file_size_mb, file_duration_in_ms

    @staticmethod
    def transcribe_audio(audio_filepath: str = None, model: str = "whisper-1"):
        """
        :param audio_filepath:
        :param model:
        :return:
        """
        audio = open(audio_filepath, "rb")
        audio_transcript = openai.Audio.transcribe(model, audio)
        audio_transcript = audio_transcript.get('text')
        return audio_transcript

    @staticmethod
    def convert_milliseconds(milliseconds: float = None):
        """

        :param milliseconds:
        :return:
        """
        seconds = milliseconds // 1000
        minutes = seconds // 60
        hours = minutes // 60

        # Calculate the remaining seconds and minutes
        seconds %= 60
        minutes %= 60

        # Format the time components with leading zeros if necessary
        formatted_time = "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)
        return formatted_time

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
                        audio_chunks_files, file_size_mb, file_duration_in_ms = self.generate_audio_chunks(
                            audio_filepath=tmp_audio_save_path, max_audio_size=25, tmp_dir=tmp_audio_dir,
                            process_bar=process_audio_bar
                        )
                        st.markdown(f"""
                        #### Audio Meta:
                        - Audio file size: {round(file_size_mb, 2)} MB
                        - Audio file duration: {self.convert_milliseconds(int(file_duration_in_ms))}
                        """)

                    # Get transcript
                    start = time.time()
                    full_audio_transcript = []
                    with st.spinner('Transcribing audio. Please wait.'):
                        transcribe_audio_bar = st.progress(0, text="Transcribing...")
                        total_chunks = len(audio_chunks_files)
                        pct_cmp = [i / total_chunks for i in range(1, total_chunks + 1)]
                        for index, i in enumerate(audio_chunks_files):
                            transcript = self.transcribe_audio(i)
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
                    self.remove_temp_files(tmp_audio_dir)


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
