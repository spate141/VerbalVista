import os
from openai import OpenAI
from typing import Literal, Any


class TextToSpeech:

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def text_to_speech(
            self, model_name: str = "tts-1",
            voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = "echo",
            text: str = "",
            save_file_path: Any = None
    ):
        response = self.client.audio.speech.create(
            model=model_name,
            voice=voice,
            input=text
        )
        return response
        # response.stream_to_file(save_file_path)

