import os
import numpy as np
from openai import OpenAI
from typing import Dict, List
from dotenv import load_dotenv; load_dotenv(".env")
from utils import log_info, log_debug, log_error


def get_embedding_client(embedding_model_name: str) -> OpenAI:
    """
    Initializes an embedding client based on the specified embedding model name. Supports OpenAI's API for
    text embeddings.

    :param embedding_model_name: The name of the embedding model to initialize. This function currently supports
                                 OpenAI's text-embedding models.
    :return: An instance of OpenAI's API client configured for embedding tasks.
    """
    if "text-embedding" in embedding_model_name:
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class EmbedChunks:
    """
    A class to manage the embedding of text chunks using a specified model. This class is designed to be
    called with batches of text, generating embeddings for each batch and logging the process.
    """
    def __init__(self, model_name: str):
        """
        Initializes the EmbedChunks instance with a specific model for embedding generation.

        :param model_name: The name of the model to be used for generating embeddings.
        """
        self.model_name = model_name
        self.embedding_client = get_embedding_client(embedding_model_name=self.model_name)
        self.counter = 1

    def __call__(self, batch: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        """
        Generates embeddings for a batch of text, incrementing the counter for each batch processed.

        :param batch: A dictionary containing 'text' and 'source' keys. 'text' should be a list of strings for which
                      embeddings will be generated. 'source' should be a list of sources corresponding to each text.
        :return: A dictionary with the original 'text' and 'source', along with 'embeddings'—the generated embeddings
                 for each text, and 'embedding_model'—the model used for embedding, repeated for each text in the batch.
        """
        response = self.embedding_client.embeddings.create(input=batch["text"], model=self.model_name)
        embeddings = np.array([i.embedding for i in response.data])
        log_debug(f"Generated embeddings for document-{self.counter}: {np.array(embeddings).shape}")
        self.counter += 1
        return {
            "text": batch["text"], "source": batch["source"],
            "embeddings": embeddings, "embedding_model": np.array([self.model_name] * len(batch["text"]))
        }

