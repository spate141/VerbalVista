import base64
import requests
from typing import Optional, Dict, Any


class OpenAIGPT4ImageAnalysisUtil:
    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initializes the OpenAI GPT-4 Image Analysis utility with the provided API key.

        :param api_key: The API key for authenticating requests to OpenAI's API.
        """
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    @staticmethod
    def encode_image(image_path: str) -> str:
        """
        Encodes the given image to a base64 string.

        :param image_path: The filesystem path to the image file.
        :return: The base64-encoded representation of the image.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_image(
            self, image, prompt: str = None, max_tokens: int = 300, streamlit_origin: bool = False
    ) -> Dict[str, Any]:
        """
        Sends an image to OpenAI's API for analysis, using GPT-4 Turbo and returns the response.

        :param image: The filesystem path to the image file or bytes.
        :param prompt: User's query or question about uploaded image.
        :param max_tokens: The maximum number of tokens to generate in the response.
        :param streamlit_origin: If the file is coming from Streamlit tool or not.
        :return: The API response as a dictionary.
        """
        if not streamlit_origin:
            base64_image = self.encode_image(image)
        else:
            base64_image = image
        payload = {
            "model": "gpt-4-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": max_tokens
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)
        return response.json()

