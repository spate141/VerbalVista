import os
import openai
import requests
from datetime import datetime
from .logging_module import log_debug


class ImageGeneration:

    def __init__(self):
        pass

    @staticmethod
    def generate_image(
            prompt: str = None,
            image_size: str = None,
            images_to_generate: int = None,
            generated_images_dir: str = None
    ):

        # call the OpenAI API
        generation_response = openai.Image.create(
            prompt=prompt,
            n=images_to_generate,
            size=image_size,
            response_format="url",
        )
        log_debug(f'Image generated')
        img_tag = prompt.replace(' ', '')[:10]
        datetime_obj = datetime.fromtimestamp(generation_response['created'])
        formatted_date_time = datetime_obj.strftime("%Y-%m-%d_%H-%M-%S")
        generated_image_filepaths = []
        for i in range(images_to_generate):
            generated_image_name = f"{img_tag}_{i}_{formatted_date_time}.png"
            generated_image_filepath = os.path.join(generated_images_dir, generated_image_name)
            generated_image_url = generation_response["data"][i]["url"]
            generated_image = requests.get(generated_image_url).content
            with open(generated_image_filepath, "wb") as f:
                f.write(generated_image)
            log_debug(f'Image saved: {generated_image_filepath}')
            generated_image_filepaths.append(generated_image_filepath)
        return generated_image_filepaths

