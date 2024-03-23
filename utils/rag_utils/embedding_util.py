import numpy as np
from langchain_openai import OpenAIEmbeddings
from utils import log_info, log_debug, log_error


def get_embedding_model(embedding_model_name):
    """
    Given the embedding_model_name; this will either use the OpenAI API or
    download the model with HuggingFaceEmbeddings.
    :param embedding_model_name: Model name, could also be model_path
    :return embedding model class instance
    """
    if "text-embedding" in embedding_model_name:
        embedding_model = OpenAIEmbeddings(
            model=embedding_model_name
        )
        return embedding_model


class EmbedChunks:

    def __init__(self, model_name):
        self.model_name = model_name
        self.embedding_model = get_embedding_model(
            embedding_model_name=self.model_name
        )

    def __call__(self, batch):
        embeddings = self.embedding_model.embed_documents(batch["text"])
        log_debug(f"Generated embeddings shape: {np.array(embeddings).shape}")
        return {
            "text": batch["text"], "source": batch["source"],
            "embeddings": embeddings, "embedding_model": np.array([self.model_name] * len(embeddings))
        }

