import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv; load_dotenv(".env")
from utils import log_info, log_debug, log_error


def get_embedding_client(embedding_model_name):
    """
    Given the embedding_model_name; this will either use the OpenAI API or
    download the model with HuggingFaceEmbeddings.
    :param embedding_model_name: Model name, could also be model_path
    :return embedding model class instance
    """
    if "text-embedding" in embedding_model_name:
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class EmbedChunks:

    def __init__(self, model_name):
        self.model_name = model_name
        self.embedding_client = get_embedding_client(embedding_model_name=self.model_name)
        self.counter = 1

    def __call__(self, batch):
        response = self.embedding_client.embeddings.create(input=batch["text"], model=self.model_name)
        embeddings = np.array([i.embedding for i in response.data])
        log_debug(f"Generated embeddings for document-{self.counter}: {np.array(embeddings).shape}")
        self.counter += 1
        return {
            "text": batch["text"], "source": batch["source"],
            "embeddings": embeddings, "embedding_model": np.array([self.model_name] * len(batch["text"]))
        }

