import os
import time
from openai import OpenAI
from pydub import AudioSegment
from pydub.utils import mediainfo
from typing import List, Dict, Optional, Any, Tuple
from utils import log_info, log_debug, log_error


class OpenAIWisperUtil:

    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initializes the OpenAI Whisper utility with the provided API key.

        :param api_key: The API key for authenticating requests to OpenAI's Whisper service.
        """
        self.client = OpenAI(api_key=api_key)

    @staticmethod
    def load_audio_data(filepath: str) -> Dict[str, Any]:
        """
        Loads audio data from the specified file path and retrieves its metadata including size in MB and duration in milliseconds.

        :param filepath: The path to the audio file.
        :return: A dictionary containing the loaded audio data, file size in MB, and file duration in milliseconds.
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
    def create_a_file(directory: str, filename: str) -> str:
        """
        Creates a file in the specified directory with the given filename. Creates the directory if it does not exist.

        :param directory: The directory where the file will be created.
        :param filename: The name of the file to create.
        :return: The path to the created file.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, filename)
        return file_path

    @staticmethod
    def split_sizes_into_chunks(actual_size: float, max_size: int) -> List[float]:
        """
        Splits an actual size in MB into a list of chunk sizes in MB, each less than or equal to the max size.

        :param actual_size: The actual size of the audio file in MB.
        :param max_size: The maximum size of each chunk in MB.
        :return: A list of chunk sizes in MB.
        """
        chunks = []
        remaining = actual_size
        while remaining > 0:
            chunk_size = min(max_size, remaining)
            chunks.append(chunk_size)
            remaining -= chunk_size
        return chunks

    def get_audio_chunks_duration(self, file_size_mb: float, file_duration_in_ms: float, max_audio_size: int = 25) -> List[float]:
        """
        Calculates the duration in milliseconds for each audio chunk based on the file size in MB and the total duration.

        :param file_size_mb: The size of the audio file in MB.
        :param file_duration_in_ms: The duration of the audio file in milliseconds.
        :param max_audio_size: The maximum size of each audio chunk in MB.
        :return: A list of durations in milliseconds for each audio chunk.
        """
        # split the original audio file into appropriate chunks of size <= max_audio_size
        file_chunks_sizes_mb = self.split_sizes_into_chunks(actual_size=file_size_mb, max_size=max_audio_size)

        # find the audio duration of each chunk based on the audio chunk size in MB
        file_chunks_durations_ms = [(i * file_duration_in_ms) / file_size_mb for i in file_chunks_sizes_mb]
        # assert sum(file_chunks_durations_ms) == file_duration_in_ms
        return file_chunks_durations_ms

    def generate_audio_chunks(self, audio_filepath: str, max_audio_size: int = 25, tmp_dir: str = 'tmp_dir/') -> Tuple[List[str], float, float]:
        """
        Generates audio chunks from the specified audio file, each chunk having a maximum size in MB.

        :param audio_filepath: The path to the audio file.
        :param max_audio_size: The maximum size of each audio chunk in MB.
        :param tmp_dir: The directory where temporary audio chunks will be stored.
        :return: A tuple containing a list of paths to the generated audio chunks, the original file size in MB, and the original file duration in milliseconds.
        """
        # start = time.time()
        audio_meta = self.load_audio_data(audio_filepath)
        audio = audio_meta['audio']
        file_size_mb = audio_meta['file_size_mb']
        file_duration_in_ms = audio_meta['file_duration_in_ms']

        file_chunks_durations_ms = self.get_audio_chunks_duration(
            file_size_mb=file_size_mb,
            file_duration_in_ms=file_duration_in_ms,
            max_audio_size=max_audio_size
        )
        # total_chunks = len(file_chunks_durations_ms)
        # pct_cmp = [i / total_chunks for i in range(1, total_chunks + 1)]
        audio_chunks_files = []
        for index, chunk_duration_ms in enumerate(file_chunks_durations_ms, 1):
            output_file = self.create_a_file(tmp_dir, f"tmp_audio_chunk_{index}.mp3")
            log_debug(f"Audio chunk created: {output_file}")
            audio_chunk = audio[:chunk_duration_ms]
            audio_chunk.export(output_file, format="mp3")
            audio = audio[chunk_duration_ms:]
            audio_chunks_files.append(output_file)
            # process_bar.progress(pct_cmp[index - 1], f'Audio processed: {round(time.time() - start, 2)} sec')
        return audio_chunks_files, file_size_mb, file_duration_in_ms

    def transcribe_audio(self, audio_filepath: str, model: str = "whisper-1") -> Optional[str]:
        """
        Transcribes the specified audio file using OpenAI's Whisper model.

        :param audio_filepath: The path to the audio file to transcribe.
        :param model: The name of the Whisper model to use for transcription.
        :return: The transcribed text, or None if transcription fails.
        """
        audio = open(audio_filepath, "rb")
        response = self.client.audio.transcriptions.create(model=model, file=audio)
        # dict_keys(['text', 'task', 'language', 'duration', 'words'])
        audio_transcript_obj = response.model_dump()
        return audio_transcript_obj.get("text", None)

    @staticmethod
    def convert_milliseconds(milliseconds: float) -> str:
        """
        Converts a duration in milliseconds to a formatted string in the format HH:MM:SS.

        :param milliseconds: The duration in milliseconds.
        :return: A string representing the formatted duration.
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

