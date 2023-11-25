import os
import requests
from openai import OpenAI
from datetime import datetime
from utils import log_debug


class OpenAIDalleUtil:

    def __init__(self, api_key=None):
        self.client = OpenAI(api_key=api_key)

    def generate_image(
            self,
            prompt: str = None,
            image_size: str = None,
            images_to_generate: int = None,
            generated_images_dir: str = None,
            model_name: str = "dall-e-3"
    ):

        # call the OpenAI API
        generation_response = self.client.images.generate(
            model=model_name,
            prompt=prompt,
            n=images_to_generate,
            size=image_size,
            response_format="url",
        )
        log_debug(f'Image generated')
        img_tag = prompt.replace(' ', '')[:10]
        datetime_obj = datetime.fromtimestamp(generation_response.created)
        formatted_date_time = datetime_obj.strftime("%Y-%m-%d_%H-%M-%S")
        generated_image_filepaths = []
        for i in range(images_to_generate):
            generated_image_name = f"{img_tag}_{i}_{formatted_date_time}.png"
            generated_image_filepath = os.path.join(generated_images_dir, generated_image_name)
            generated_image_url = generation_response.data[i].url
            generated_image = requests.get(generated_image_url).content
            with open(generated_image_filepath, "wb") as f:
                f.write(generated_image)
            log_debug(f'Image saved: {generated_image_filepath}')
            generated_image_filepaths.append(generated_image_filepath)
        return generated_image_filepaths
