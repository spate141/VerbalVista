import os
import requests
from openai import OpenAI
from datetime import datetime
from typing import List, Optional, Literal
from utils import log_debug


class OpenAIDalleUtil:

    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initializes the OpenAI DALL·E utility with the provided API key.

        :param api_key: The API key for authenticating requests to OpenAI's DALL·E service.
        """
        self.client = OpenAI(api_key=api_key)

    def generate_image(
        self,
        prompt: str = None,
        image_size: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"] = "1024x1024",
        generated_images_dir: str = None,
        model_name: str = "dall-e-3",
        image_quality: Literal["standard", "hd"] = "standard"
    ) -> str:
        """
        Generates images based on the given prompt using OpenAI's DALL·E and saves them to the specified directory.

        :param prompt: The prompt to generate images for.
        :param image_size: The size of the generated images, one of the specified resolutions.
        :param generated_images_dir: The directory to save the generated images.
        :param model_name: The name of the DALL·E model to use. Defaults to "dall-e-3".
        :param image_quality: Image quality. standard or hd.
        :return: A filepath for the generated image.
        """

        # call the OpenAI API
        generation_response = self.client.images.generate(
            model=model_name, prompt=prompt, n=1, size=image_size, response_format="url", quality=image_quality
        )
        log_debug(f'Image generated')
        img_tag = prompt.replace(' ', '')[:10]
        datetime_obj = datetime.fromtimestamp(generation_response.created)
        formatted_date_time = datetime_obj.strftime("%Y-%m-%d_%H-%M-%S")
        generated_image_name = f"{img_tag}_{formatted_date_time}.png"
        generated_image_filepath = os.path.join(generated_images_dir, generated_image_name)
        generated_image_url = generation_response.data[0].url
        generated_image = requests.get(generated_image_url).content
        with open(generated_image_filepath, "wb") as f:
            f.write(generated_image)
        log_debug(f'Image saved: {generated_image_filepath}')
        return generated_image_filepath

