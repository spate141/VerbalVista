import os
import time
from openai import OpenAI
from pydub import AudioSegment
from pydub.utils import mediainfo
from utils import log_info, log_debug, log_error


class OpenAIWisperUtil:

    def __init__(self, api_key=None):
        self.client = OpenAI(api_key=api_key)

    @staticmethod
    def load_audio_data(filepath):
        """

        :param filepath:
        :return:
        """
        # load audio
        extension = os.path.splitext(filepath)[-1][1:]
        audio = AudioSegment.from_file(filepath, format=extension)
        log_debug(f"Audio file read into memory: {filepath}")

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
        # assert sum(file_chunks_durations_ms) == file_duration_in_ms
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
            process_bar.progress(pct_cmp[index - 1], f'Audio processed: {round(time.time() - start, 2)} sec')
        return audio_chunks_files, file_size_mb, file_duration_in_ms

    def transcribe_audio(self, audio_filepath: str = None, model: str = "whisper-1"):
        """
        :param audio_filepath:
        :param model:
        :return:
        """
        audio = open(audio_filepath, "rb")
        audio_transcript = self.client.audio.transcriptions.create(model=model, file=audio)
        return audio_transcript.text

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

