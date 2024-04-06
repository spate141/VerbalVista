from openai import OpenAI
from typing import Literal, Any, Optional


class OpenAIText2SpeechUtil:

    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initializes the OpenAI text-to-speech utility with the provided API key.

        :param api_key: The API key for authenticating requests to OpenAI's text-to-speech service.
        """
        self.client = OpenAI(api_key=api_key)

    def text_to_speech(
        self,
        text: str,
        model_name: str = "tts-1",
        voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = "echo",
        save_file_path: Optional[str] = None
    ) -> Any:
        """
        Generates speech from text using OpenAI's text-to-speech service and optionally saves it to a file.

        :param text: The text to convert to speech.
        :param model_name: The name of the text-to-speech model to use. Defaults to "tts-1".
        :param voice: The voice to use for speech generation. Defaults to "echo".
        :param save_file_path: The file path to save the generated speech audio. If None, the audio will not be saved.
        :return: The response from the OpenAI text-to-speech service.
        """
        response = self.client.audio.speech.create(
            model=model_name,
            voice=voice,
            input=text
        )
        return response
        # response.stream_to_file(save_file_path)

